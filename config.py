class Configs():
    def __init__(self):
        # 1 2 3 5 (0 1 2 4)
        self.classes = 5
        self.input_channels = 12
        self.model_hidden = [128, 256, 512, 1024, 4096]
        self.Transformer_dmodel = 256