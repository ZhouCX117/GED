import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import yaml
import argparse
import torch
import torchvision
from tqdm.auto import tqdm
from accelerate import Accelerator, DistributedDataParallelKwargs
from utils import *
from data_multifuse_select8 import *
from torch.utils.data import DataLoader
from fvcore.common.config import CfgNode
from accelerate.utils import set_seed
from accelerate.state import AcceleratorState

import accelerate
from diffusers import (
    UNet2DConditionModel,
    AutoencoderKL
)
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers
from shutil import copyfile


train_caption=open("/data/zhoucaixia/Dataset/BSDS/minigpt4_caption_train.txt",'r').readlines()

train_caption_dict={}
for key in range(len(train_caption)):
    str_line=train_caption[key].strip("\n").split("\t")
    train_caption_dict[str_line[0]]=str_line[1]

test_caption=open("/data/zhoucaixia/Dataset/BSDS/minigpt4_caption_test.txt",'r').readlines()

test_caption_dict={}
for key in range(len(test_caption)):
    str_line=test_caption[key].strip("\n").split("\t")
    test_caption_dict[str_line[0]]=str_line[1]



def parse_args():
    parser = argparse.ArgumentParser(description="training vae configure")
    parser.add_argument("--cfg", help="experiment configure file name", type=str, default='/data/zhoucaixia/workspace/DiffusionEdge/configs/BSDS_train.yaml')
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default='/data/zhoucaixia/LM/stable-diffusion-2-1',
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument( 
        "--image_size",
        type=int,
        default=[320,320]
    )
    # parser.add_argument("")
    args = parser.parse_args()
    args.cfg = load_conf(args.cfg)
    return args

def load_conf(config_file, conf={}):
    with open(config_file) as f:
        exp_conf = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in exp_conf.items():
            conf[k] = v
    return conf


def main(args):
    cfg = CfgNode(args.cfg)
    torch.manual_seed(42)
    np.random.seed(42)
    set_seed(42)
    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]
    
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path,
                                            subfolder='vae')
        text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path,
                                                     subfolder='text_encoder')
        
        unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path,subfolder="unet",
                                                    in_channels=4, sample_size=320/8,time_cond_proj_dim=320,
                                                    low_cpu_mem_usage=False,
                                                    ignore_mismatched_sizes=True)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.train() 
    vae=vae.cuda()
    text_encoder=text_encoder.cuda()
    unet=unet.cuda()
    
    data_cfg = cfg.data

    if data_cfg['name'] == 'edge':
        dataset = EdgeDataset(
            data_root=data_cfg.img_folder,
            image_size=args.image_size,
            augment_horizontal_flip=data_cfg.augment_horizontal_flip,
            cfg=data_cfg
        )
    else:
        raise NotImplementedError
    dl = DataLoader(dataset, batch_size=data_cfg.batch_size, shuffle=True, pin_memory=True,
                    num_workers=data_cfg.get('num_workers', 2))
    
    test_dataset = EdgeDatasetTest(
            data_root='/data/zhoucaixia/Dataset/BSDS/images/test',
            image_size=args.image_size,
        )
    test_dl = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True,
                    num_workers=1)

    
    train_cfg = cfg.trainer
    
    os.makedirs(train_cfg.results_folder,exist_ok=True)
    runfile_name=os.path.basename(__file__)
    copyfile(os.path.join("/data/zhoucaixia/workspace/DiffusionEdge",runfile_name),os.path.join(train_cfg.results_folder,runfile_name))
    copyfile("/data/zhoucaixia/workspace/DiffusionEdge/configs/BSDS_train.yaml",os.path.join(train_cfg.results_folder,'BSDS_train.yaml'))
    copyfile("/data/zhoucaixia/workspace/DiffusionEdge/denoising_diffusion_pytorch/data_multifuse_select8.py",os.path.join(train_cfg.results_folder,'data_multifuse_select8.py'))
    
    trainer = Trainer(
        vae,text_encoder, unet,dl,test_dl, train_batch_size=data_cfg.batch_size,
        gradient_accumulate_every=train_cfg.gradient_accumulate_every,
        train_lr=train_cfg.lr, train_num_steps=train_cfg.train_num_steps,
        save_and_sample_every=train_cfg.save_and_sample_every, results_folder=train_cfg.results_folder,
        amp=train_cfg.amp, fp16=train_cfg.fp16, log_freq=train_cfg.log_freq, cfg=cfg,
        resume_milestone=train_cfg.resume_milestone,
        train_wd=train_cfg.get('weight_decay', 1e-4)
    )
    trainer.train()
    pass


class Trainer(object):
    def __init__(
            self,
            vae,
            text_encoder,
            unet,
            data_loader,
            test_data_loader,
            train_batch_size=16,
            gradient_accumulate_every=1,
            train_lr=1e-4,
            train_wd=1e-4,
            train_num_steps=100000,
            save_and_sample_every=1000,
            num_samples=25,
            results_folder='./results',
            amp=False,
            fp16=False,
            split_batches=True,
            log_freq=20,
            resume_milestone=0,
            cfg={},
    ):
        super().__init__()
        ddp_handler = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision='fp16' if fp16 else 'no',
            kwargs_handlers=[ddp_handler],
        )
        self.enable_resume = cfg.trainer.get('enable_resume', False)
        self.accelerator.native_amp = amp
        accelerator = self.accelerator
        device = accelerator.device
        self.vae=vae
        self.text_encoder=text_encoder
        self.unet=unet

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.log_freq = log_freq

        self.train_num_steps = train_num_steps

        dl = self.accelerator.prepare(data_loader)
        self.dl = cycle(dl)
        
        
        test_dl = self.accelerator.prepare(test_data_loader)
        self.test_dl = test_dl
        
        for k,v in self.unet.named_parameters():
            if 'time' in k or 'up_blocks.3' in k or 'up_blocks.2' in k or 'conv_out' in k or 'conv_norm' in k:
                v.requires_grad=True
            else:
                v.requires_grad=False
                
        self.opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.unet.parameters()),
                                     lr=train_lr, weight_decay=train_wd)

        lr_lambda = lambda iter: max((1 - iter / train_num_steps) ** 0.96, cfg.trainer.min_lr)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda=lr_lambda)
        # for logging results in a folder periodically
        if self.accelerator.is_main_process:
            self.results_folder = Path(results_folder)
            self.results_folder.mkdir(exist_ok=True, parents=True)
            self.ema = EMA(self.unet, ema_model=None, beta=0.999,
                           update_after_step=cfg.trainer.ema_update_after_step,
                           update_every=cfg.trainer.ema_update_every)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.vae,self.text_encoder,self.unet, self.opt, self.lr_scheduler = \
            self.accelerator.prepare(self.vae,self.text_encoder,self.unet, self.opt, self.lr_scheduler)
        self.logger = create_logger(root_dir=results_folder)
        self.logger.info(cfg)
        self.results_folder = Path(results_folder)

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        
        self.tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path,subfolder='tokenizer')
        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:
                total_loss = 0.
                
                for ga_ind in range(self.gradient_accumulate_every):
                    batch = next(self.dl)
                    for key in batch.keys():
                        if isinstance(batch[key], torch.Tensor):
                            batch[key].to(device)

                    with self.accelerator.autocast():
                        edge_list=batch['image']
                        gran_list=batch['gran']
                        
                        image = batch['cond'] if 'cond' in batch else None
                        edge_torch,gran_torch,image_torch=[],[],[]
                        for index_x in range(edge_list[0].shape[0]):
                            edge_torch.append(edge_list[0][index_x].unsqueeze(0).unsqueeze(0))
                            gran_torch.append(gran_list[index_x])
                            image_torch.append(image)
                        edge_torch=torch.cat(edge_torch,0)
                        gran_torch=torch.cat(gran_torch)
                        image_torch=torch.cat(image_torch)
                        
                        with torch.no_grad():
                            edge_stacked=edge_torch.repeat(1,3,1,1)
                            h_rgb = self.vae.encoder(image_torch)
                            moments_rgb = self.vae.quant_conv(h_rgb)
                            mean_rgb, logvar_rgb = torch.chunk(moments_rgb, 2, dim=1)
                            rgb_latents = mean_rgb *0.18215
                            
                            
                            h_edge = self.vae.encoder(edge_stacked)
                            moments_edge = self.vae.quant_conv(h_edge)
                            mean_edge, logvar_edge = torch.chunk(moments_edge, 2, dim=1)
                            edge_latents = mean_edge * 0.18215
                        
                        bsz = edge_latents.shape[0]
                        timesteps = torch.ones((bsz,), device=edge_latents.device)
                        timesteps = timesteps.long()
                        
                        
                        
                        prompt = train_caption_dict[batch['img_name'][0]]
                       
                        text_inputs =self.tokenizer(
                            prompt,
                            padding="do_not_pad",
                            max_length=self.tokenizer.model_max_length,
                            truncation=True,
                            return_tensors="pt",
                        )
                        
                        text_input_ids = text_inputs.input_ids.cuda() #[1,2]
                            
                        text_embeds = self.text_encoder(text_input_ids)[0].cuda()
                        batch_empty_text_embed = text_embeds .repeat((edge_latents.shape[0], 1, 1))
                            
                        target=edge_latents
                        
                        
                        z_pred = self.unet(rgb_latents, 
                                        timesteps, 
                                        encoder_hidden_states=batch_empty_text_embed,timestep_cond=gran_torch.view(-1,1)).sample  # [B, 4, h, w]
                        
                        sample_distances = torch.mean((z_pred[:,None, :,  :, :] - z_pred[None,:,  :, :, :])**2, dim=(2,3, 4))
                        gt_sample_distances = torch.mean((target[:,None, :,  :, :] - target[None,:,  :, :, :])**2, dim=(2,3, 4))
                        
                        
                        loss_diff = F.mse_loss(sample_distances, gt_sample_distances.detach(), reduction="mean")
                        
                        with torch.no_grad():
                            z=self.vae.post_quant_conv(z_pred/0.18215)
                            stacked = self.vae.decoder(z)
                            edge_mean = stacked.mean(dim=1, keepdim=True)
                            edge_mean = (edge_mean + 1.0) / 2.0
                            torchvision.utils.save_image(edge_mean,self.results_folder/'1.jpg')
                        
                        edge_mean_sum=torch.sum(edge_mean,dim=(1,2,3))
                        predict_gran_normalization=(edge_mean_sum-batch['min_num'])/(batch['max_num']-batch['min_num'])
                        loss_gran = F.mse_loss(predict_gran_normalization,gran_torch,reduction="mean")
                        
                        loss_mse = F.mse_loss(z_pred.float(), target.float(), reduction="mean")
                        loss = (loss_mse+loss_diff+loss_gran) / self.gradient_accumulate_every
                        total_loss += loss.item()
                        torch.cuda.empty_cache()
                        
                    self.accelerator.backward(loss)
                
                describtions = "[Train Step] {}/{},loss_mean:{},loss_diff:{},loss_gran:{}".format(self.step, self.train_num_steps,loss_mse,loss_diff,loss_gran) 
                if accelerator.is_main_process:
                    pbar.desc = describtions

                if self.step % self.log_freq == 0:
                    if accelerator.is_main_process:
                        self.logger.info(describtions)
                        

                accelerator.clip_grad_norm_(filter(lambda p: p.requires_grad, self.unet.parameters()), 1.0)
                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()
                self.lr_scheduler.step()
                

                accelerator.wait_for_everyone()

                self.step += 1
                
                
                if accelerator.is_main_process:
                    self.ema.to(device)
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        milestone = self.step // self.save_and_sample_every
                        # self.save(milestone)
                        save_path = os.path.join(self.results_folder, f"checkpoint-{milestone}")
                        accelerator.save_state(save_path)
                        with torch.no_grad():
                            self.unet.eval()
                            for idx, batch in tqdm(enumerate(self.test_dl)):
                                for key in batch.keys():
                                    if isinstance(batch[key], torch.Tensor):
                                        batch[key].to(device)
                                cond = batch['cond']
                                img_name = batch["img_name"][0]
                                
                                prompt=test_caption_dict[img_name]
                                gran=torch.ones((1,), device=cond.device)
                                for i_g in [0,1]:
                                    grant=gran*i_g
                                    batch_pred = self.slide_sample(cond, crop_size=[320, 320], stride=[240, 240], prompt=prompt,gran=grant)
                                    file_name = self.results_folder / str(self.step//self.save_and_sample_every)/str(i_g)/img_name
                                    os.makedirs(self.results_folder / str(self.step//self.save_and_sample_every)/str(i_g),exist_ok=True)
                                    torchvision.utils.save_image(batch_pred, str(file_name)[:-4] + ".png")

                accelerator.wait_for_everyone()
                pbar.update(1)
                self.unet.train()
        accelerator.print('training complete')

    def slide_sample(self, inputs, crop_size, stride, prompt,gran):
    
        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = 1
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        aux_out1 = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]
                device=inputs.device
                
                h_rgb = self.vae.encoder(crop_img)
                moments_rgb = self.vae.quant_conv(h_rgb)
                mean_rgb, logvar_rgb = torch.chunk(moments_rgb, 2, dim=1)
                rgb_latents = mean_rgb *0.18215
               
                text_inputs =self.tokenizer(
                    prompt,
                    padding="do_not_pad",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids.cuda() #[1,2]
                
                empty_text_embed = self.text_encoder(text_input_ids)[0].cuda()
                
                t = torch.ones( (rgb_latents.shape[0],), device=rgb_latents.device)
                
                z_pred = self.unet(
                    rgb_latents, t, encoder_hidden_states=empty_text_embed,timestep_cond=gran.view(-1,1)
                ).sample

            
                torch.cuda.empty_cache()
                edge_latent=z_pred/0.18215
                z=self.vae.post_quant_conv(edge_latent)
                stacked = self.vae.decoder(z)
                # mean of output channels
                edge_mean = stacked.mean(dim=1, keepdim=True)
                
                edge = (edge_mean + 1.0) / 2.0
                
                crop_seg_logit = edge
                
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))
    
                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        
        seg_logits = preds / count_mat
        aux_out1 = aux_out1 / count_mat
        
        
        return seg_logits

if __name__ == "__main__":
    args = parse_args()
    main(args)
    pass