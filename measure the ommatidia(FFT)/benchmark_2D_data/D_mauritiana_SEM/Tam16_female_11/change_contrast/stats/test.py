import pstats
from pstats import SortKey
p = pstats.Stats('intensity_1.000')
p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(30)
