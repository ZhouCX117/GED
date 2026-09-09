Code for "Generative Edge Detection with Stable Diffusion"

Please modify the source code and explicitly add a projection module:

```
self.cond_proj = nn.Sequential(

nn.Linear(1, 320),

nn.GELU(),

nn.Linear(320, 320),
)
```
