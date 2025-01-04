# SetTrajectory.msg
string action          # hover, generate_sin, generate_line, generate_polynomial, generate_point_stabilizing, generate_circle, load
int32 duration        # duration in seconds
string file_path      # optional path to trajectory file, used when action is 'load'