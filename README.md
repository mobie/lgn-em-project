# LGN-EM

Access the data:
- Install MoBIE-beta, following [these instructions](https://github.com/mobie/mobie-viewer-fiji/blob/master/MOBIE-BETA.md). 
- Store the aws credentials so that the dataset can be loaded. For this, follow [these instructions](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html).
- Now, open the Fiji with MoBIE-beta and select `Plugins->MoBIE->Open MoBIE Project ...`.
- Enter `https://github.com/mobie/lgn-em-datasets` in the prompt that opens.
- This should open a few menus and a viewer which shows the EM data.

Viewing the neuron segmentation:
- Select `segmentation->neurons` and click `view` in the MoBIE GUI (the menu with the whale logo)
- The current segmentation covers only a small part of the dataset, which is not part of the inital view. To find it, enter `{"position":[326.64837813452584,119.43274034618811,31.013969544771427]}` in `location` and click on `move`.

Saving interesting positions:
- You can save interesting positions by right click on a position in the viewer and then selecting `BDV - Log Current View`.
- This will print the position and view to the `Log` window; see below for an example output.
- The position can be recovered by pasting the line containing `position` (only changes position) or `normalizedAffine` (recvoers the full view) into the `location` menu.

```
{"position":[326.64837813452584,119.43274034618811,31.013969544771427],"timepoint":0}
{"affineTransform":[8.655308093335417,0.0,0.0,-2390.2423509426494,0.0,8.655308093335417,0.0,-744.2271641275895,0.0,0.0,8.655308093335417,-268.4354616073183],"timepoint":0}
{"normalizedAffine":[0.009903098504960432,0.0,0.0,-3.2348310651517727,0.0,0.009903098504960432,0.0,-1.182754192365663,0.0,0.0,0.009903098504960432,-0.3071343954317143],"timepoint":0}
```
