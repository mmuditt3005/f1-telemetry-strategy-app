import fastf1
fastf1.Cache.enable_cache('f1_cache')

try:
    session = fastf1.get_session(2025, 'China', 'Q')
    session.load()

    laps = session.laps
    ham = laps.pick_driver('HAM').pick_fastest()
    ver = laps.pick_driver('VER').pick_fastest()

    print("✅ China 2025 loaded!")
    print(f"Total laps in session: {len(laps)}")
    print(f"HAM lap time: {ham['LapTime']}")
    print(f"VER lap time: {ver['LapTime']}")

except Exception as e:
    print("❌ Could not load 2025 China qualifying.")
    print("Reason:", str(e))
