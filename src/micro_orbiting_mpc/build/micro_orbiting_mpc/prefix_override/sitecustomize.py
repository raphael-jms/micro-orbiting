import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/raphael/micro-orbiting/src/micro_orbiting_mpc/install/micro_orbiting_mpc'
