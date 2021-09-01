# NeedleInsertionModule (RPMB system) #
A slicer module for visualizing (and executing) best entry points for robotic needle insertion.

In pathology and legal medicine, the histopathological and microbiological analysis of tissue samples from infected deceased is a valuable information for developing treatment strategies during a pandemic such as COVID-19.
However, a conventional autopsy carries the risk of disease transmission and may be rejected by relatives.
We propose minimally invasive biopsy with robot assistance under CT guidance to minimize the risk of disease transmission during tissue sampling and to improve accuracy.
A flexible robotic system for biopsy sampling is presented, which is applied to human corpses placed inside protective body bags.
An automatic planning and decision system estimates optimal insertion point.
Heat maps projected onto the segmented skin visualize the distance and angle of insertions and estimate the minimum cost of a puncture while avoiding bone collisions.
Further, we test multiple insertion paths concerning feasibility and collisions.
A custom end effector is designed for inserting needles and extracting tissue samples under robotic guidance.
Our robotic post-mortem biopsy (RPMB) system is evaluated in a study during the COVID-19 pandemic on 20 corpses and 10 tissue targets, 5 of them being infected with \linebreak SARS-CoV-2.
The mean planning time including robot path planning is 5.72±1.67 s. Mean needle placement accuracy is 7.19±4.22 mm.


## Prerequisites ##
1. Slicer > 4.11

This module is mostly tested with Slicer 4.11.

## How To Install ##
No installation needed. Just tell Slicer where to find the module.

1. Open *Edit->Application Settings*
2. Go to Tab *Modules*
3. Add the path */path/to/BestPathVisualization* to *Additional module paths*
4. (Optionall) Clone the robot communication server and dependencies

The robot communication server depends on a custom version of the KUKA Fri library. Therefore, we cannot freely distribute it. Please contact us directly if you own a KUKA manipulator in order to acquire the modified library.

## How To Use ##
1. Import a CT scan
2. Open the *BestPathVisualization* module
3. (Optionally) Connect to the robot control server and load a transformation matrix from robot base to CT coordinate system
4. Select Volume, Output Model, and Markup Target
5. Calculate Maximum Density for a colored density map
6. (Optionally) Execute the planned insertion from surface point to target

## How To Cite ##
ToDo
