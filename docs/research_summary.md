# Research Summary

## Core Direction

The recurring message across recent panoramic perception work is that image
content should be modeled on the sphere. For oriented multi-object tracking,
this means image rotation, annotation rotation, detector training, and tracker
association should all respect spherical geometry.

## Paper Notes

### Towards Oriented Multi-object Tracking for Fisheye Images

This paper motivates the target problem directly: fisheye scenes distort object
shape and direction, so axis-aligned boxes are not enough. The tracking system
needs oriented annotations, oriented detection, oriented association, and an
evaluation protocol based on rotated overlap.

### SO3UFormer

SO3UFormer studies rotation-robust panoramic segmentation by learning intrinsic
spherical features. The useful idea for this project is the SO(3) treatment of
panoramic signals. A panoramic augmentation should rotate rays on the sphere,
not rotate the 2D raster as if it were a flat image.

### PanoVGGT

PanoVGGT applies feed-forward 3D reconstruction to panoramic imagery and uses
spherical-aware geometric representations. For tracking, its relevance is the
same: object centers and box boundaries are better represented as viewing rays
or longitude/latitude coordinates than as plain image-plane coordinates.

### PanDA

PanDA targets panoramic depth estimation. Depth is not required for the first
OBB tracking baseline, but it is a strong second-stage cue for resolving ID
switches under occlusion, scale changes, and seam crossings.

### ViT3

ViT3 focuses on test-time training for vision transformers. In this project it
is an optional adaptation layer for an OBB detector or segmenter when test
panoramas have rotations or distortions not seen during training.

### PriOr-Flow

PriOr-Flow improves panoramic optical flow by combining primitive panoramic
flow and orthogonal-view reasoning. Its most useful tracking role is short-term
box or polygon propagation when detector confidence drops.

## Engineering Principle

Start with the geometry and data format. A detector or tracker cannot fix labels
that were rotated incorrectly. The first milestone should therefore be a
verified spherical image and label rotation tool.

