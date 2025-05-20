from .syn_and_connec_construct import build_populations as populations
from .syn_and_connec_construct import build_up_recurrent_connections as recurrent_connections
from .syn_and_connec_construct import build_fixed_aff_to_pop_matrix as fixed_aff_to_pop_matrix
from .syn_and_connec_construct import build_fixed_afference as fixed_afference
from .syn_and_connec_construct import initialize_to_rest,\
        initialize_to_random, get_syn_and_conn_matrix,\
        random_distance_dependent_connections,\
        draw_spatially_dependent_connectivity_profile, random_connections
from .syn_and_connec_library import *
