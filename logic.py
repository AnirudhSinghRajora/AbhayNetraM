import time
import threading
import requests
from fastapi import FastAPI

app = FastAPI()


current_green_timings = None
cycle_end_time = None
lock = threading.Lock()


CYCLE_DURATION = 120    
OBSERVATION_DURATION = 5  
STREAM_IDS = [0, 1, 2, 3]  
MIN_GREEN = 15
MAX_GREEN = 55

def fetch_density(stream_id):
    try:
        response = requests.get(f"http://localhost:8000/density/{stream_id}", timeout=1)
        if response.status_code == 200:
            data = response.json()
            return data.get("density", 0)
    except Exception as e:
        print(f"Error fetching density for stream {stream_id}: {e}")
    return 0

def compute_green_timings(avg_densities, total_cycle=120, min_time=15, max_time=55):
    stream_ids = list(avg_densities.keys())
    total_density = sum(avg_densities.values())
    
    
    if total_density == 0:
        return {sid: total_cycle / len(stream_ids) for sid in stream_ids}
    
    
    raw_allocations = {
        sid: (avg_densities[sid] / total_density) * total_cycle for sid in stream_ids
    }
    
    
    times = {
        sid: max(min(raw_allocations[sid], max_time), min_time) for sid in stream_ids
    }
    
    
    S = sum(times.values())
    delta = total_cycle - S
    
    
    for _ in range(20):
        if abs(delta) < 1e-2:
            break
        if delta > 0:
            
            active = [sid for sid in stream_ids if times[sid] < max_time]
            if not active:
                break
            addition = delta / len(active)
            for sid in active:
                times[sid] = min(times[sid] + addition, max_time)
        else:  
            active = [sid for sid in stream_ids if times[sid] > min_time]
            if not active:
                break
            subtraction = delta / len(active)
            for sid in active:
                times[sid] = max(times[sid] + subtraction, min_time)
        S = sum(times.values())
        delta = total_cycle - S
    
    return times

def update_green_timings():
    global current_green_timings, cycle_end_time
    while True:
        cycle_start_time = time.time()
        density_samples = {sid: [] for sid in STREAM_IDS}
        observation_end = cycle_start_time + OBSERVATION_DURATION

        
        while time.time() < observation_end:
            for sid in STREAM_IDS:
                density_samples[sid].append(fetch_density(sid))
            time.sleep(1)

        
        avg_densities = {
            sid: (sum(density_samples[sid]) / len(density_samples[sid]) if density_samples[sid] else 0)
            for sid in STREAM_IDS
        }
        
        
        timings = compute_green_timings(avg_densities, total_cycle=CYCLE_DURATION, min_time=MIN_GREEN, max_time=MAX_GREEN)
        
        with lock:
            current_green_timings = timings
            cycle_end_time = cycle_start_time + CYCLE_DURATION

        
        print("Cycle started at:", cycle_start_time)
        print("Avg densities:", avg_densities)
        print("Green light timings:", timings)
        print("Cycle will end in {:.1f} seconds".format(cycle_end_time - time.time()))
        
        
        time_remaining = cycle_end_time - time.time()
        if time_remaining > 0:
            time.sleep(time_remaining)

@app.get("/green_timings")
def get_green_timings():
    with lock:
        if current_green_timings is None:
            return {"message": "Green timings not set yet."}
        remaining = cycle_end_time - time.time() if cycle_end_time is not None else 0
        return {
            "green_timings": current_green_timings,
            "cycle_remaining": max(0, round(remaining, 1))
        }

if __name__ == "__main__":
    
    thread = threading.Thread(target=update_green_timings, daemon=True)
    thread.start()
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
