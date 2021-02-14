#!/bin/bash    

while getopts f:t: option
do
        case "${option}"
        in
                f) FILE=${OPTARG};;
                t) TOPIC_ROOT=${OPTARG};;
        esac
done

set -x
echo "vt_id.dat"
rostopic echo -b ratslam_out.bag -p /irat_red/LocalView/Template > vt_id.dat
echo "em_id.dat"
rostopic echo -b ratslam_out.bag -p /irat_red/PoseCell/TopologicalAction > em_id.dat
echo "pose.dat"
rostopic echo -b ratslam_out.bag -p /irat_red/ExperienceMap/RobotPose > pose.dat
echo "map.dat"
rostopic echo -b ratslam_out.bag -p /irat_red/ExperienceMap/Map > map.dat

