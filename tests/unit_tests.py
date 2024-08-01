import sys
import unittest
from unit_test_helpers import *
from matplotalt import *

# Line
# Bar
# Scatter
# Area
# Pie
# Radial line
# Strip
# Contour
# Heatmap
# Boxplot

from matplotalt.matplotalt_constants import *

# Exact string descriptions may change, so we focus on testing internal representations

# Do ChartDescriptions capture the correct data?

# Do they capture the correct labels?

# Do they raise errors when input is malformed?

# ---------------------------------------------------

# Are chart types detected correctly?

class TestChartDescMethods(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chart_func_to_desc_object = {}
        for chart_func in CHART_FUNC_TO_DATA.keys():
            chart_func()
            ax = infer_single_axis()
            chart_type = infer_chart_type(ax)
            self.chart_func_to_desc_object[chart_func] = CHART_TYPE_TO_CLASS[chart_type](ax)
            plt.clf()


    def assertOrderedDictEqual(self, d1, d2):
        d1l = list(d1.items())
        d2l = list(d2.items())
        self.assertEqual(len(d1l), len(d2l))
        for i in range(len(d1l)):
            self.assertEqual(d1l[i][0],  d2l[i][0])
            if isinstance(d1l[i][1], (np.ndarray, list)):
                if len(d1l[i][1]) > 0 and  isinstance(d1l[i][1][0], (np.ndarray, list)):
                    for j in range(len(d1l[i][1])):
                        curd1_arr = np.array(d1l[i][1][j])
                        if curd1_arr.dtype in [np.int32, np.float64]:
                            np.testing.assert_allclose(d1l[i][1][j], d2l[i][1][j])
                        else:
                            np.testing.assert_equal(d1l[i][1][j], d2l[i][1][j])
                else:
                    curd1_arr = np.array(d1l[i][1])
                    if curd1_arr.dtype in [np.int32, np.float64]:
                        np.testing.assert_allclose(d1l[i][1], d2l[i][1])
                    else:
                        np.testing.assert_equal(d1l[i][1], d2l[i][1])
            else:
                self.assertEqual(d1l[i][1],  d2l[i][1])


    def test_desc_chart_types(self):
        for chart_func, chart_data in CHART_FUNC_TO_DATA.items():
            if chart_data["type"] is not None:
                chart_desc_obj = self.chart_func_to_desc_object[chart_func]
                self.assertEqual(chart_data["type"], chart_desc_obj.get_chart_type())


    def test_desc_data(self):
        for chart_func, chart_data in CHART_FUNC_TO_DATA.items():
            if chart_data["data"] is not None:
                chart_desc_obj = self.chart_func_to_desc_object[chart_func]
                self.assertOrderedDictEqual(chart_data["data"], chart_desc_obj.get_axes_data())


    def test_desc_ticklabels(self):
        for chart_func, chart_data in CHART_FUNC_TO_DATA.items():
            if chart_data["ticklabels"] is not None:
                chart_desc_obj = self.chart_func_to_desc_object[chart_func]
                self.assertOrderedDictEqual(chart_data["ticklabels"], chart_desc_obj.get_axes_ticklabels())


    def test_desc_axislabels(self):
        for chart_func, chart_data in CHART_FUNC_TO_DATA.items():
            if chart_data["axislabels"] is not None:
                chart_desc_obj = self.chart_func_to_desc_object[chart_func]
                self.assertOrderedDictEqual(chart_data["axislabels"], chart_desc_obj.get_axes_labels())


    def test_desc_axistypes(self):
        for chart_func, chart_data in CHART_FUNC_TO_DATA.items():
            if chart_data["axistypes"] is not None:
                chart_desc_obj = self.chart_func_to_desc_object[chart_func]
                self.assertOrderedDictEqual(chart_data["axistypes"], chart_desc_obj.get_axes_types())



if __name__ == '__main__':
    unittest.main()