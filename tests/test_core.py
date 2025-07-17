import numpy as np
from thermal_model import ThermalManagementModel


def test_temperature_rises_under_charge():
    model = ThermalManagementModel(
        capacity=50,
        internal_resistance=0.1,
        ambient_temperature=25,
        num_zones=3,
    )
    history = model.simulate(current=5, time_steps=5, verbose=False)
    # Expect shape (time_steps, num_zones)
    assert history.shape == (5, 3)
    # Temperatures should not be lower than ambient at any time for positive current
    assert np.all(history >= 25) 


def test_external_heat_increases_temp():
    model = ThermalManagementModel(
        capacity=50,
        internal_resistance=0.1,
        ambient_temperature=25,
        num_zones=3,
    )
    # No current but external heat applied
    history = model.simulate(current=0, time_steps=3, verbose=False, external_heat=50)
    assert np.all(history[-1] > 25)  # temperature should rise above ambient 


def test_contact_resistance_influences_conduction():
    model_lowR = ThermalManagementModel(
        capacity=50,
        internal_resistance=0.1,
        ambient_temperature=25,
        num_zones=2,
        contact_resistance=0.001,
    )
    model_highR = ThermalManagementModel(
        capacity=50,
        internal_resistance=0.1,
        ambient_temperature=25,
        num_zones=2,
        contact_resistance=0.01,
    )
    # Apply same external heat to zone 0 only via manual temperature diff
    model_lowR.temperature = np.array([35.0, 25.0])
    model_highR.temperature = np.array([35.0, 25.0])

    q_low = model_lowR._zone_heat_losses(0)
    q_high = model_highR._zone_heat_losses(0)
    # Higher contact resistance implies lower conduction heat loss magnitude
    assert abs(q_high) < abs(q_low) 