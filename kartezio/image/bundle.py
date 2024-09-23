from kartezio.image.nodes import IMAGE_NODES_ABBV_LIST
from kartezio.model.components import KartezioBundle


class BundleOpenCV(KartezioBundle):
    def fill(self):
        for node_abbv in IMAGE_NODES_ABBV_LIST:
            self.add_node(node_abbv)


BUNDLE_OPENCV = BundleOpenCV()
