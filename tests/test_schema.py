from src.schema import DataCfg, ModelCfg, GACfg

def test_schema_defaults():
    d = DataCfg(target="y")
    m = ModelCfg()
    g = GACfg()
    assert d.test_size==0.2 and d.random_state==42
    assert m.C==1.0 and m.max_iter==200
    assert g.population_size>0 and g.generations>0
