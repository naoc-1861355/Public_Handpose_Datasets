# Public_Handpose_Datasets
This repository contains code in python for exploring and visualizing public handpose datasets listed below.<br>
Most datasets contains code to convert them to standardized format (ezxr format)

## Datasets included
<table>
    <tr>
        <th>Institution</th>
        <th>dataset</th>
        <th>size</th>
    </tr>
    <tr>
        <td rowspan="4">CMU</td>
        <td>Hands manual</td>
        <td>2.7k</td>
    </tr>
    <tr>
        <td>Hands from Synthetic Data</td>
        <td>14k</td>
    </tr>
    <tr>
        <td>Hands from Panoptic Studio</td>
        <td>14k</td>
    </tr>
    <tr>
        <td>mtc</td>
        <td>111k</td>
    </tr>
    <tr>
        <td rowspan="2">Freiburg</td>
        <td>FreiHand dataset</td>
        <td>32k * 4</td>
    </tr>
    <tr>
        <td>Rendered Handpose Dataset (RHD)</td>
        <td>43k</td>
    </tr>
    <tr>
        <td>City University of HK</td>
        <td>STB</td>
        <td>18k</td>
    </tr>
        <tr>
        <td>Ariel AI</td>
        <td>YouTube 3D Hands</td>
        <td>50k</td>
    </tr>
        <tr>
        <td>University of Alicante</td>
        <td>MHP</td>
        <td>80k</td>
    </tr>
        <tr>
        <td rowspan="2">MPII</td>
        <td>GANerated Hands</td>
        <td>330k</td>
    </tr>
    <tr>
        <td>SynthHands</td>
        <td>63k</td>
    </tr>
    <tr>
        <td>University of Minnesota</td>
        <td>HUMBI</td>
        <td>24M</td>
    </tr>
    <tr>
        <td>Seoul National University</td>
        <td>InterHand2.6M</td>
        <td>2.4M</td>
    </tr>
</table>

## script included
`read_draw.py` provide script to read and visualize ezxr formatted dataset <br>

for each dataset, the following scripts are included:

  - `viz.py` visualize key points annotation <br>
  - `normdat.py` convert datasets to same ezxr format
  
## ezxr format
In ezxr format of datasets, each rgb image corresponding to a json annotation in the following format
```
    -- f02-f42: thumb to little finger in 2d
        -- each finger from finger palm to tip
    -- f52: wrist in 2d
    -- f03-f43: thumb to little finger in 3d
        -- each finger from finger palm to tip
    -- f52: wrist in 3d
    -- is_left: 1-left, 0-right, -1-unknown
    -- palm_center: palm center of hand in 3d
    -- hand_bbox: hand bounding box in 2d [x_min, x_max, y_min, y_max]
```


