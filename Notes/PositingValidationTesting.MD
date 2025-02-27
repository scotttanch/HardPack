# Positing System Validation and Charecterization
We have various positing systems: UWB, RealSense, IMU, and Combinations therein,
but no good way to evaluate their accuracy beyond tape and pencil. To achieve
this we will use the CNC machine. This will allow movement between known positions
with 0.1 mm accuracy [Double Check That (seems to be true)]. 

Using Vcarve 8 we will create toolpaths with known geometry, and are also able to
control the feedrate (or speed) at which the machine moves along the toolpath.

Alternatively, paths can be created in gcode to begin with.

## Data Collection
The general test plan is to position the HardPack ontop of the router head and
have the cnc extecute several toolpaths while recording data with a variety of
sensors.
### Part 1: RealSense Only
This step is simple, run all the paths with just the RealSense

### Part 2: RealSense and IMU
Enable the IMU and Record its data in addtion to the RealSense

## Data Processing
1. Raw RealSense
2. Filtered RealSense
3. Filtereed RealSense and IMU

# Preliminary Results

