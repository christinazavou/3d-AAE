from unittest import TestCase
from contentstylecomponent import ContentStyleComponentDataset

import open3d as o3d
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


class TestAnnfassComponentDataset(TestCase):
    def test_me(self):
        inp_path = "/media/graphicslab/BigData/zavou/ANNFASS_CODE/"
        inp_path += "style_detection/logs/annfass_content_style_splits"
        dset = ContentStyleComponentDataset(inp_path)
        for i in range(5):
            content_xyz, content_detailed_xyz, style_xyz, _ = dset.__getitem__(i)
            content_pcd = o3d.geometry.PointCloud()
            content_detailed_pcd = o3d.geometry.PointCloud()
            style_pcd = o3d.geometry.PointCloud()
            content_pcd.points = o3d.utility.Vector3dVector(content_xyz)
            content_detailed_pcd.points = o3d.utility.Vector3dVector(content_detailed_xyz)
            style_pcd.points = o3d.utility.Vector3dVector(style_xyz)
            o3d.visualization.draw_geometries([content_pcd.translate([-0.5, 0, 0]),
                                               content_detailed_pcd,
                                               style_pcd.translate([0.5, 0, 0])],)
