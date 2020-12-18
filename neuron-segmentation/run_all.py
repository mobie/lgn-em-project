import os
prep = __import__('1_prepare_table')
pred = __import__('2_predict_boundaries')
prob = __import__('3_problem')
lift = __import__('4_lifted_problems_from_boutons')
lmc = __import__('5_lmc')


def run_all(prepare=False):
    gpus = list(range(4))
    if prepare:
        prep.prepare_table()
    # pred.predict_boundaries_3d(gpus)
    prob.set_up_problem()

    # NOTE annotation table and format might change
    annotation_table = os.path.join('/g/rompani/lgn-em-datasets/data/0.0.0/tables/sbem-adult-1-lgn-boutons',
                                    'bouton_annotations_v1_done.csv')
    keep_annotations = ['merge', 'fragment', 'bouton']
    lift.make_lifted_problem(annotation_table, keep_annotations)

    lmc.run_lmc()


if __name__ == '__main__':
    run_all()
