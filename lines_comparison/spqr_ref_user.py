# https://github.com/Vahe1994/SpQR/blob/902abdb012b24b07d0f4376bdc46952e4189add1/inference_demo.py
@contextmanager
def suspend_nn_inits():
    def skip(*args, **kwargs):
        pass

    saved_inits = torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_  # saving
    torch.nn.init.kaiming_uniform_ = torch.nn.init.uniform_ = torch.nn.init.normal_ = skip  # replacing
    try:
        yield
    finally:
        torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_ = saved_inits  # restoring

with suspend_nn_inits():
    with torch.no_grad():
        self.config = AutoConfig.from_pretrained(
            pretrained_model_path, torchscript=self.torchscript, return_dict=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_path,
            trust_remote_code=True,
            torch_dtype=torch.half,
            config=self.config,
        )