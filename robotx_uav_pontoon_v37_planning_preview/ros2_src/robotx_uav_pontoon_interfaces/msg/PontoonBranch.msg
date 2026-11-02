# index is image-order label 1..4, NOT a verified physical bay identifier.
uint8 index
string label
geometry_msgs/Point32 tip_uv_px
geometry_msgs/Point32 root_uv_px
geometry_msgs/Point32 center_uv_px
geometry_msgs/Point tip_world_m
geometry_msgs/Point root_world_m
geometry_msgs/Point center_world_m
bool tip_world_valid
bool root_world_valid
bool center_world_valid
