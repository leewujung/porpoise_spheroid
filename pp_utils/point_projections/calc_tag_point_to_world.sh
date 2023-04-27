mkdir output/world_locs
mkdir output/world_locs/tag_locations
mkdir output/world_locs_unrefracted
mkdir output/world_locs_unrefracted/tag_locations
printf  "Do transformation using hula calibration\n"
python -u tag_points_to_world.py -s -op output/world_locs/ -tf data/transformation_to_origin/transformation_to_origin_hula.txt
mv output/world_locs/tag_locations output/world_locs/tag_locations_hula
mkdir output/world_locs/tag_locations
printf "\nDo transformation using cross calibration\n"
python -u tag_points_to_world.py -s -op output/world_locs/ -tf data/transformation_to_origin/transformation_to_origin_cross.txt
mv output/world_locs/tag_locations output/world_locs/tag_locations_cross
printf "\nDo unrefracted transformation using hula calibration\n"
python -u tag_points_to_world.py -s -dr -op output/world_locs_unrefracted/ -tf data/transformation_to_origin/transformation_to_origin_hula.txt
mv output/world_locs_unrefracted/tag_locations output/world_locs_unrefracted/tag_locations_hula
printf "\nDo unrefracted transformation using cross calibration\n"
mkdir output/world_locs_unrefracted/tag_locations
python -u tag_points_to_world.py -s -dr -op output/world_locs_unrefracted/ -tf data/transformation_to_origin/transformation_to_origin_cross.txt
mv output/world_locs_unrefracted/tag_locations output/world_locs_unrefracted/tag_locations_cross
read -p "Press enter to continue"
