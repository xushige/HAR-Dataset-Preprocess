

//////////////////////////////////////////////
// Data collection Specification
//////////////////////////////////////////////


**********************************************
Section 1: Device Configuration
**********************************************

1. Device Type: MotionNode
2. Sampling rate: 100Hz
3. Accelerometer range: +-6g
4. Gyroscope range: +-500dps


**********************************************
Section 2: Data Format
**********************************************

Each activity trial is stored in an .mat file.

The naming convention of each .mat file is defined as:
a"m"t"n".mat, where
"a" stands for activity
"m" stands for activity number
"t" stands for trial
"n" stands for trial number
 
Each .mat file contains 13 fields:
1. title: USC Human Motion Database
2. version: it is version 1.0 for this first round data collection
3. date
4. subject number
5. age
6. height
7. weight
8. activity name
9. activity number
10. trial number
11. sensor_location
12. sensor_orientation
13. sensor_readings

For sensor_readings field, it consists of 6 readings:
From left to right:
1. acc_x, w/ unit g (gravity)
2. acc_y, w/ unit g
3. acc_z, w/ unit g
4. gyro_x, w/ unit dps (degrees per second)
5. gyro_y, w/ unit dps
6. gyro_z, w/ unit dps


**********************************************
Section 3: Activities
**********************************************

1. Walking Forward
2. Walking Left
3. Walking Right
4. Walking Upstairs
5. Walking Downstairs
6. Running Forward
7. Jumping Up
8. Sitting
9. Standing
10. Sleeping
11. Elevator Up
12. Elevator Down



