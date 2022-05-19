from tensorboard import program

tracking_address = './tensorboard/two_panda_reach_plate_joints_dense_v1'


if __name__ == "__main__":
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', tracking_address])
    url = tb.launch()
    print(f"tensorboard on {url}")

    while True:
        pass
