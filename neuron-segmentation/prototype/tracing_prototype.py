import numpy as np
import vigra
import h5py
from elf.segmentation.watershed import stacked_watershed
from elf.segmentation.multicut import multicut_gaec, compute_edge_costs
from elf.segmentation.lifted_multicut import lifted_multicut_gaec


def make_superpixels():
    with h5py.File('./test_data.h5', 'a') as f:
        bd = f['boundaries'][:]
        ws = stacked_watershed(bd, threshold=.25, sigma_seeds=2.)[0]
        ds = f.require_dataset('watershed', shape=ws.shape, dtype=ws.dtype, compression='gzip')
        ds[:] = ws
    # with napari.gui_qt():
    #     viewer = napari.Viewer()
    #     viewer.add_image(bd)
    #     viewer.add_image(ws)


def assign_superpixels_to_boutons():
    from nifty.ground_truth import overlap
    with h5py.File('./test_data.h5', 'r') as f:
        ws = f['watershed'][:]
        # boutons = f['boutons_corrected'][:]
        boutons = f['boutons'][:]
    boutons, _, _ = vigra.analysis.relabelConsecutive(boutons)

    ws_ids = np.unique(ws)
    ovlp = overlap(ws, boutons)
    ovlp = np.array([ovlp.overlapArrays(ws_id, True)[0][0] if ws_id in ws_ids else 0
                     for ws_id in range(int(ws.max() + 1))], dtype='uint32')

    with h5py.File('./test_data.h5', 'a') as f:
        if 'bouton_overlaps' in f:
            del f['bouton_overlaps']
        f.create_dataset('bouton_overlaps', data=ovlp)


def compute_graph_and_weights(path, return_edge_sizes=False):
    from nifty.graph import undirectedGraph
    with h5py.File(path, 'a') as f:
        # if 'features' in f:
        if False:
            edges = f['edges'][:]
            feats = f['features'][:]
            edge_sizes = f['edge_sizes'][:]
            z_edges = f['z_edges'][:]
            n_nodes = int(edges.max()) + 1

        else:
            from elf.segmentation.features import compute_rag, compute_boundary_features, compute_z_edge_mask
            seg = f['watershed'][:]
            boundaries = f['boundaries'][:]
            boundaries[boundaries > .2] *= 3
            boundaries = np.clip(boundaries, 0, 1)
            rag = compute_rag(seg)
            n_nodes = rag.numberOfNodes
            feats = compute_boundary_features(rag, boundaries)
            feats, edge_sizes = feats[:, 0], feats[:, -1]
            edges = rag.uvIds()

            z_edges = compute_z_edge_mask(rag, seg)

            # f.create_dataset('edges', data=edges)
            # f.create_dataset('edge_sizes', data=edge_sizes)
            # f.create_dataset('features', data=feats)
            # f.create_dataset('z_edges', data=z_edges)

    graph = undirectedGraph(n_nodes)
    graph.insertEdges(edges)
    if return_edge_sizes:
        return graph, feats, edge_sizes, z_edges, boundaries
    else:
        return graph, feats


def trace_axons_prototype(graph, feats, bouton_labels, threshold):
    from affogato.segmentation import graph_watershed_with_threshold
    assert graph.numberOfNodes == len(bouton_labels)
    edges = graph.uvIds()
    node_labels = graph_watershed_with_threshold(edges, feats, bouton_labels, threshold)
    return node_labels


def trace_from_boutons(threshold):
    path = './test_data.h5'
    with h5py.File(path, 'r') as f:
        bouton_labels = f['bouton_overlaps'][:]

    print("Computing graph and features ...")
    graph, feats = compute_graph_and_weights(path)
    print("... done")

    axon_labels = trace_axons_prototype(graph, feats, bouton_labels,
                                        threshold=threshold)

    with h5py.File('./test_data.h5', 'r') as f:
        ws = f['watershed'][:]

    import nifty.tools as nt
    seg = nt.take(axon_labels, ws)
    return seg


def check_results(seg, boundaries):
    import napari
    import nifty.tools as nt
    with h5py.File('./test_data.h5', 'r') as f:
        raw = f['raw'][:]
        # boundaries = f['boundaries'][:]
        ws = f['watershed'][:]
        # boutons = f['boutons_corrected'][:]
        b_overlaps = f['bouton_overlaps'][:]

    overlaps_mapepd = nt.take(b_overlaps, ws)

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(raw)
        viewer.add_image(boundaries, visible=False)
        viewer.add_labels(ws, visible=False)
        # viewer.add_labels(boutons, visible=False)
        viewer.add_labels(overlaps_mapepd)
        viewer.add_labels(seg)


def _merge_bouton_labels(node_labels, path):
    with h5py.File(path, 'r') as f:
        bouton_labels = f['bouton_overlaps'][:]

    merged_labels = np.zeros_like(node_labels)
    bouton_ids = np.unique(bouton_labels)
    assert len(bouton_labels) == len(node_labels)
    for bid in bouton_ids[1:]:
        node_values = np.unique(node_labels[bouton_labels == bid])
        merged_labels[np.isin(node_labels, node_values)] = bid

    return merged_labels


def run_mc(merge_boutons, beta=.7):
    path = './test_data.h5'
    graph, feats, sizes, z_edges, boundaries = compute_graph_and_weights(path, return_edge_sizes=True)
    costs = compute_edge_costs(feats, edge_sizes=sizes, weighting_scheme='z',
                               z_edge_mask=z_edges, beta=beta)
    print("Running multicut ...")
    node_labels = multicut_gaec(graph, costs)
    print("... done")

    if merge_boutons:
        node_labels = _merge_bouton_labels(node_labels, path)

    with h5py.File('./test_data.h5', 'r') as f:
        ws = f['watershed'][:]

    import nifty.tools as nt
    seg = nt.take(node_labels, ws)
    return seg, boundaries


def _lifted_problem(graph, costs, path):
    repulsive_cost = np.min(costs) - .1
    with h5py.File(path, 'r') as f:
        bouton_labels = f['bouton_overlaps'][:]

    lifted_uvs = []
    lifted_costs = []

    bouton_ids = np.unique(bouton_labels)[1:]
    for b_a in bouton_ids:
        ids_a = np.where(bouton_labels == b_a)[0]
        for b_b in bouton_ids:
            if b_b <= b_a:
                continue
            ids_b = np.where(bouton_labels == b_b)[0]

            # cartesian product
            this_edges = np.transpose([np.tile(ids_a, len(ids_b)), np.repeat(ids_b, len(ids_a))])

            local_edges = graph.findEdges(this_edges)
            local_edge_mask = local_edges != -1
            local_edges = local_edges[local_edge_mask]
            costs[local_edges] = repulsive_cost

            this_lifted = this_edges[~local_edge_mask]
            this_costs = np.ones(len(this_lifted)) * repulsive_cost

            lifted_uvs.append(this_lifted)
            lifted_costs.append(this_costs)

    lifted_uvs = np.concatenate(lifted_uvs, axis=0)
    lifted_costs = np.concatenate(lifted_costs, axis=0)
    return costs, lifted_uvs, lifted_costs


def run_lmc(merge_boutons, beta=.7):
    path = './test_data.h5'
    graph, feats, sizes, z_edges, boundaries = compute_graph_and_weights(path, return_edge_sizes=True)
    costs = compute_edge_costs(feats, edge_sizes=sizes, weighting_scheme='z',
                               z_edge_mask=z_edges, beta=beta)

    costs, lifted_uvs, lifted_costs = _lifted_problem(graph, costs, path)

    print("Running lifted multicut ...")
    node_labels = lifted_multicut_gaec(graph, costs, lifted_uvs, lifted_costs)
    print("... done")

    if merge_boutons:
        node_labels = _merge_bouton_labels(node_labels, path)

    with h5py.File('./test_data.h5', 'r') as f:
        ws = f['watershed'][:]

    import nifty.tools as nt
    seg = nt.take(node_labels, ws)
    return seg, boundaries


if __name__ == '__main__':
    # make_superpixels()
    # assign_superpixels_to_boutons()

    # seg = trace_from_boutons(threshold=.125)
    # seg, boundaries = run_mc(True, beta=.7)
    seg, boundaries = run_lmc(False, beta=.7)

    check_results(seg, boundaries)
