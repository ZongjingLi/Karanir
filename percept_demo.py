from karanir.logger import get_logger, set_output_file

set_output_file("logs/output_expr")

logger = get_logger("MLK Expr")

logger.critical("Build the model")
logger.warning("this is a custom warning.")
logger.critical("What is the shape of that?")