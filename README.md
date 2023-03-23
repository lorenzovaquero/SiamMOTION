# SiamMOTION
Real-time siamese multiple object tracker with enhanced proposals.


# Weights

Weights can be downloaded at [This link](https://drive.google.com/drive/folders/1YnhzIyJLdTXfkTb36r3wDLhZkrxnGJou?usp=share_link).

# Usage

Build and use the Docker container to solve all the dependencies. A sample code can be found in the `scripts` folder:

```
cd Docker
./run_SiamMT.sh
cd SiamMOTION
python ./scripts/track.py './demo-sequence/vot15_bag/imgs' --detailed --nologs -p ./model/SiamMOTION/parameters.json
```


# References

```
@article{VaqueroSiamMOTION,
  author    = {Lorenzo Vaquero and
               V{\'{\i}}ctor M. Brea and
               Manuel Mucientes},
  title     = {Real-time siamese multiple object tracker with enhanced proposals},
  journal   = {Pattern Recognition},
  volume    = {135},
  pages     = {109141},
  year      = {2023}
}
```


## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.
