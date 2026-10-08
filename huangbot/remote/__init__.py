"""Play HuangBot in the Zhanguo web game through its External Bot Protocol v1.

The subpackage is optional and independent of OpenSpiel. All Zhanguo-specific
knowledge lives here; ``model.py``, ``checkpoints.py`` and the weights are
unchanged. Import the submodules you need (``huangbot.remote.cli`` is the entry
point) so that importing this package stays cheap.
"""

__version__ = "0.1"

__all__ = ["__version__"]
