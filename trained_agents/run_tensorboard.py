from tensorboard import program

# tracking_address = './tensorboard/panda_reach_v2'
# tracking_address = './tensorboard/My_PandaReachJointsDense'
# tracking_address = './tensorboard/panda_push_v2'
# tracking_address = './tensorboard/two_reach_dense_v1'
# tracking_address = './tensorboard/three_reach_dense_v1'
# tracking_address = './tensorboard/my_panda_pick_and_place_v1'
# tracking_address = './tensorboard/panda_pick_and_place_v2'
# tracking_address = './tensorboard/panda_pick_and_place_dense_v2'
# tracking_address = './tensorboard/my_panda_slide_dense_v1'
# tracking_address = './tensorboard/panda_reach_plate_joints_dense_v1'
# tracking_address = './tensorboard/two_panda_reach_plate_joints_dense_v1'
# tracking_address = './tensorboard/two_obj_push_dense_v1'
# tracking_address = './tensorboard/two_push_dense_v1'
tracking_address = './tensorboard/two_pick_and_place_v1'

if __name__ == "__main__":
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', tracking_address])
    url = tb.launch()
    print(f"tensorboard on {url}")

    while True:
        pass
