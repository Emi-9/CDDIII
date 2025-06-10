import pytest
import numpy as np
from analisis.descriptivo import AnalisisDescriptivo

@pytest.fixture
def datos():
    return [1.0, 2.5, 3.7, 2.0, 5.0]

def test_evalua_histograma_con_lista(datos):
    ad = AnalisisDescriptivo(datos)
    x = [1.5, 2.5, 3.5]
    h = 1
    res = ad.evalua_histograma(h, x)
    assert isinstance(res, np.ndarray)
    assert len(res) == len(x)
    # valores en cada bin conocidos manualmente
    assert np.isclose(res[1], pytest.approx(0.2))  # proporción en el bin

def test_evalua_histograma_con_array(datos):
    ad = AnalisisDescriptivo(np.array(datos))
    x = np.array([2.0, 4.0])
    res = ad.evalua_histograma(0.5, x)
    assert res.dtype == float

def test_bins_comportamiento(datos):
    ad = AnalisisDescriptivo(datos)
    h = 2
    bins = np.arange(min(datos) - h, max(datos) + h + 1, h)
    x = [min(datos), max(datos)]
    res = ad.evalua_histograma(h, x)
    assert len(res) == 2

def test_error_si_no_data():
    ad = AnalisisDescriptivo([])
    with pytest.raises(AssertionError):
        ad.evalua_histograma(1, [1.0])
