# ade20k Dataset Exploration

## Running Notebooks

Notebooks are stored as `.py` files (jupytext percent format). To run one:

```bash
uv run jupytext --sync notebooks/<filename>.py
```

## Dataset

**Name:** ade20k

| Resource | Link |
|----------|------|
| Paper    | <!-- e.g. https://arxiv.org/abs/... --> |
| GitHub   | <!-- e.g. https://github.com/... --> |
| HuggingFace | <!-- e.g. https://huggingface.co/datasets/... --> |

## Properties of Interest

- Object size distribution
- Object location distribution
- Object thinness (area of the object over area of its convex hull)
- Object contrast with background distribution
- How the above object properties relate to each other.
- The sizes of the images (look with your eyes to see how much cropping changes the image and squishing changes the image.)


## Outputs

### `outputs/ade20k_df_flat.parquet` — one row per (image, class) pair

| Column | Type | Description |
|--------|------|-------------|
| `image_idx` | int | 0-based index into the sorted validation annotation file list |
| `class_idx` | int | 1-based ADE20K class index (1–150) |
| `class_name` | str | Primary name of the class (first comma-separated entry from objectInfo150.txt) |
| `area` | float | Fraction of image pixels occupied by the class |

### `outputs/ade20k_df_stats.parquet` — one row per class, indexed by `class_idx`

| Column | Type | Description |
|--------|------|-------------|
| `mean_area` | float | Mean per-image area fraction across images where the class appears |
| `max_area` | float | Maximum per-image area fraction |
| `min_area` | float | Minimum per-image area fraction |
| `name` | str | Primary class name |

## Experiment Context

**Project / experiment this feeds into:**
CanViT ADE20K segmentation performance on different types of objects (on diff subsets of ADE20K)

On certain tasks, you need higher output resolutions. To obtain higher output resolutions with a model like DINOV3 (B or S), you need higher image resolutions. We cannot decouple feature map/output resolution from image resolution, making image processing very computationally costly and possibly intractable. Also, you cannot recover performance if the image resolution is too low, even by using B.

However, CanViT can decouple image resolution from canvas resolution, allowing CanViT to be computationally efficient and performant on tasks requiring hight output resolutions.

**Goals:**
What do you want to understand about this dataset before using it?
- Explore the properties of the small objects in this dataset

- [ ] ...
