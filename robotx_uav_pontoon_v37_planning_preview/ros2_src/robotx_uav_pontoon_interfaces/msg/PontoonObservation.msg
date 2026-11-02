# Header is copied from the SOURCE IMAGE; frame_id MUST be camera optical frame.
std_msgs/Header header
string world_frame_id
string detector_version
string status
string invalid_reason
float32 geometry_score  # geometry fit score; NOT a probability
bool detected
bool layout_verified
bool keypoints_valid
bool geometry_ready
bool branch_ids_verified
bool uv_coordinates_verified
bool metric_calibration_verified
bool xyz_valid  # all trunk and branch points valid in world_frame_id
bool planning_ready  # geometry_ready && xyz_valid && all manual verification gates
uint8 branch_count
uint32 image_width
uint32 image_height
float32[4] bbox_xywh_px
geometry_msgs/Point32 trunk_start_uv_px
geometry_msgs/Point32 trunk_end_uv_px
geometry_msgs/Point32 trunk_center_uv_px
geometry_msgs/Point trunk_start_world_m
geometry_msgs/Point trunk_end_world_m
geometry_msgs/Point trunk_center_world_m
bool trunk_start_world_valid
bool trunk_end_world_valid
bool trunk_center_world_valid
string xyz_method  # aligned_depth, known_plane, mixed, none
robotx_uav_pontoon_interfaces/PontoonBranch[] branches
