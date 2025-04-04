import fastf1
from fastf1.plotting import setup_mpl
import matplotlib.pyplot as plt

fastf1.Cache.enable_cache('f1_cache')

def compare_drivers(year, grand_prix, session_type, driver1, driver2):
    session = fastf1.get_session(year, grand_prix, session_type)
    session.load()

    d1 = session.laps.pick_driver(driver1).pick_fastest()
    d2 = session.laps.pick_driver(driver2).pick_fastest()

    d1_tel = d1.get_telemetry()
    d2_tel = d2.get_telemetry()

    setup_mpl(misc_mpl_mods=False)
    plt.figure(figsize=(12, 5))
    plt.plot(d1_tel['Distance'], d1_tel['Speed'], label=driver1)
    plt.plot(d2_tel['Distance'], d2_tel['Speed'], label=driver2)

    plt.xlabel("Distance (m)")
    plt.ylabel("Speed (km/h)")
    plt.title(f"Fastest Lap Speed: {driver1} vs {driver2}")
    plt.legend()
    plt.tight_layout()
    graph_path = f"static/{driver1}_vs_{driver2}.png"
    plt.savefig(graph_path)
    return graph_path
