import mujoco
from mujoco import mjx

with open("./models/rodent_optimized.xml", "r") as f:
    XML_OPT = f.read()

with open("./models/rodent.xml", "r") as f:
    XML_OG = f.read()


model = mjx.put_model(mj_model_mjx)
data = mjx.make_data(model)
data = mjx.forward(model, data)

def benchmark(xml: str):
    ...
