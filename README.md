# vid2vr

This repo creates a VR180 stereoscopic video from a normal 2D video. It does this by running depth estimation on the frames and using pytorch3d to do a 3d transformation of the camera to simulate the view from each eye. 

Frames will be concatenated and the resulting video interpolated to 60fps for headsets. The output will adopt the input resolution however this should be square (and using a standard VR resolution, e.g. 2048x2048).

You will need the VR180 Creator Tool to stitch the output L/R videos into the final VR video for upload.

Serious props to https://twitter.com/gandamu_ml for writing transform_image_3d, the function used to transform for the eye images.

## Environment

Setup pytorch environment (using e.g. conda).

## Dependencies

Clone https://github.com/isl-org/MiDaS.git
Clone https://github.com/shariqfarooq123/AdaBins.git
Clone https://github.com/MSFTserver/pytorch3d-lite.git

```
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
python3 setup.py install
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)


