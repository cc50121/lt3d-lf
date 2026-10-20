import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


camera = load_script('export_camera_results')
download = load_script('download_release')


class ReleaseToolsTest(unittest.TestCase):
    def setUp(self):
        self.infos = [dict(token='token123', cams={
            view: dict(data_path='samples/{}/{}.jpg'.format(view, view))
            for view in camera.VIEWS})]
        self.coco = dict(categories=[dict(id=100 + i, name=c)
                                    for i, c in enumerate(camera.CLASSES)],
                         images=[dict(id=view, file_name=view + '.jpg')
                                 for view in camera.VIEWS])

    def test_category_names_not_category_id_offsets(self):
        with tempfile.TemporaryDirectory() as root:
            detection = dict(image_id='CAM_FRONT', category_id=109,
                             bbox=[10, 20, 30, 40], score=0.8)
            report = camera.export(self.coco, self.infos, iter([detection]), root)
            self.assertEqual(report['camera_files'], 6)
            self.assertEqual(report['detections'], 1)
            text = (Path(root) / 'token123@CAM_FRONT.txt').read_text().split()
            self.assertEqual(text, ['9', '10.0', '20.0', '40.0', '60.0', '0.8'])
            self.assertEqual((Path(root) / 'token123@CAM_BACK.txt').read_text(), '')
            with self.assertRaises(FileExistsError):
                camera.export(self.coco, self.infos, iter([]), root)

    def test_missing_camera_is_an_error(self):
        self.coco['images'].pop()
        with self.assertRaises(ValueError):
            camera.mappings(self.coco, self.infos)

    def test_nan_is_an_error(self):
        detection = dict(image_id='CAM_FRONT', category_id=100,
                         bbox=[10, 20, 30, 40], score=float('nan'))
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):
                camera.export(self.coco, self.infos, iter([detection]), root)

    def test_final_calibration_is_not_legacy_dino(self):
        data = json.loads((ROOT / 'configs/final_codino_calibration.json').read_text())
        self.assertEqual(set(data['coefficients']), set(camera.CLASSES))
        self.assertEqual(data['coefficients']['adult'], 0.1)
        self.assertEqual(data['coefficients']['barrier'], 0.216)

    def test_archive_path_escape_is_rejected(self):
        import io
        import tarfile
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'bad.tar.gz'
            with tarfile.open(path, 'w:gz') as archive:
                entry = tarfile.TarInfo('../escaped')
                entry.size = 1
                archive.addfile(entry, io.BytesIO(b'x'))
            with self.assertRaises(ValueError):
                download.unpack(path, Path(root) / 'destination')
            self.assertFalse((Path(root) / 'escaped').exists())

    def test_checkpoint_reassembly(self):
        import hashlib
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root / 'configs').mkdir()
            (root / 'downloads').mkdir()
            assets = []
            for name, data in [('weights.part001', b'abc'), ('weights.part002', b'def')]:
                (root / 'downloads' / name).write_bytes(data)
                assets.append(dict(name=name, kind='checkpoint_part',
                                   sha256=hashlib.sha256(data).hexdigest()))
            manifest = dict(assets=assets, assembled=[dict(
                name='weights.pth', parts=[a['name'] for a in assets],
                sha256=hashlib.sha256(b'abcdef').hexdigest())])
            (root / 'configs/release_assets.json').write_text(json.dumps(manifest))
            with patch.object(download, 'ROOT', root), patch('sys.argv', ['download_release.py']):
                download.main()
            self.assertEqual((root / 'downloads/weights.pth').read_bytes(), b'abcdef')


if __name__ == '__main__':
    unittest.main()
