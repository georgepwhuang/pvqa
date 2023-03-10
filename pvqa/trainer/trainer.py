import warnings

from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm import TqdmExperimentalWarning

from pvqa.constants import CONFIG_DIR, LOG_DIR
from pvqa.qencoder.interfaces import QEncoder
from pvqa.util import LocalPauliObservables


class PVQACLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_subclass_arguments(QEncoder, "qencoder")
        parser.add_class_arguments(LocalPauliObservables, "observables")
        parser.add_argument("n_qubits", type=int)
        parser.add_argument("qdim", type=int)
        parser.add_argument("locality", type=int)
        parser.link_arguments("n_qubits", "observables.qubits")
        parser.link_arguments("locality", "observables.locality")
        parser.link_arguments("qdim", "data.init_args.qdim")
        parser.link_arguments("n_qubits", "qencoder.init_args.n_qubits")
        parser.link_arguments("observables", "qencoder.init_args.observable_list", apply_on="instantiate")
        parser.link_arguments("qencoder", "data.init_args.qencoder", apply_on="instantiate")
        parser.link_arguments("data.qencode_dim", "model.init_args.input_dim", apply_on="instantiate")
        parser.link_arguments("data.init_args.one_hot", "model.init_args.multilabel", apply_on="instantiate")
        parser.set_defaults({"trainer.logger": TensorBoardLogger(save_dir=LOG_DIR)})


warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
cli = PVQACLI(parser_kwargs={"default_config_files": [str(CONFIG_DIR.joinpath("config.yaml"))]}, run=False)
cli.trainer.fit(cli.model, cli.datamodule)
cli.trainer.test(cli.model, cli.datamodule, ckpt_path="best")
