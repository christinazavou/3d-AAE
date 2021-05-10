from unittest import TestCase
from buildingcomponent import BuildingComponentDataset

import open3d as o3d
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


class TestAnnfassComponentDataset(TestCase):
    def test_me(self):
        inp_path = "/media/graphicslab/BigData/zavou/ANNFASS_CODE/"
        # inp_path += "style_detection/logs/annfass_content_style_splits/style"
        inp_path += "style_detection/logs/buildnet_content_style_splits/style"
        dset = BuildingComponentDataset(inp_path, split='train', n_points=2048)
        for i in range(5):
            xyz, _ = dset.__getitem__(i)
            print(xyz)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            o3d.visualization.draw_geometries([pcd],)
