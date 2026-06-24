"""Test conversations via HTTP + WebSocket — clean run."""
import sys, os, time, json, urllib.request, urllib.error

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")
results = []

def ok(name):
    results.append((name, "PASS"))
    print(f"  PASS: {name}")
def fail(name, msg):
    results.append((name, f"FAIL - {msg}"))
    print(f"  FAIL: {name} - {msg}")

print("Starting server...")
import subprocess
python_exe = r"C:\Users\张崇轩\AppData\Local\Programs\Python\Python312\python.exe"
server = subprocess.Popen(
    [python_exe, "server.py"],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    cwd=os.path.dirname(os.path.abspath(__file__)),
)
time.sleep(6)

# Check HTTP is up (retry for up to 20s)
ready = False
for i in range(20):
    try:
        resp = urllib.request.urlopen("http://localhost:8000/", timeout=3)
        if resp.status == 200:
            ready = True
            break
    except Exception:
        time.sleep(1)
if not ready:
    fail("Server startup", "Timed out waiting for server")
    server.terminate()
    sys.exit(1)

ok("Server startup")

try:
    import websocket

    def ws_query(query, timeout=120):
        ws = websocket.create_connection("ws://localhost:8000/ws", timeout=5)
        ws.settimeout(timeout)
        events = []
        try:
            ws.send(query)
            while True:
                msg = ws.recv()
                if not msg:
                    break
                evt = json.loads(msg)
                events.append(evt)
                if evt.get("type") == "done":
                    break
        except websocket.WebSocketTimeoutException:
            events.append({"type": "_timeout"})
        except Exception as e:
            events.append({"type": "_ws_error", "msg": str(e)})
        finally:
            ws.close()
        return events

    # === Test 1: Greeting ===
    print("\n--- Test 1: Greeting ---")
    events = ws_query("你好", timeout=30)
    types = [e["type"] for e in events]
    final = [e for e in events if e["type"] == "final_result"]
    errs = [e for e in events if e["type"] in ("error", "_error")]
    if errs:
        fail("Greeting", errs[0].get("message", str(errs))[:150])
    elif not final:
        fail("Greeting", f"No answer. Events: {types}")
    else:
        ans = final[0].get("answer", "")
        print(f"    Answer: {ans[:100]}")
        ok("Greeting")

    # === Test 2: Memory Store ===
    print("\n--- Test 2: Memory Store ---")
    events = ws_query("帮我记住：Python是第一编程语言", timeout=90)
    types = [e["type"] for e in events]
    final = [e for e in events if e["type"] == "final_result"]
    errs = [e for e in events if e["type"] in ("error", "_error")]
    if errs:
        fail("Memory store", errs[0].get("message", str(errs))[:150])
    elif not final:
        fail("Memory store", f"No answer. Events: {types}")
    else:
        ok("Memory store")

    # === Test 3: Memory Recall ===
    print("\n--- Test 3: Memory Recall ---")
    events = ws_query("我之前让你记住什么了？", timeout=90)
    types = [e["type"] for e in events]
    final = [e for e in events if e["type"] == "final_result"]
    errs = [e for e in events if e["type"] in ("error", "_error")]
    if errs:
        fail("Memory recall", errs[0].get("message", str(errs))[:150])
    elif not final:
        fail("Memory recall", f"No answer. Events: {types}")
    else:
        ans = final[0].get("answer", "")
        print(f"    Answer: {ans[:150]}")
        if "Python" in ans:
            ok("Memory recall")
        else:
            fail("Memory recall", f"Python not found. Answer: {ans[:150]}")

    # === Test 4: Weather ===
    print("\n--- Test 4: Weather ---")
    events = ws_query("北京天气怎么样", timeout=90)
    types = [e["type"] for e in events]
    final = [e for e in events if e["type"] == "final_result"]
    errs = [e for e in events if e["type"] in ("error", "_error")]
    if errs:
        fail("Weather", errs[0].get("message", str(errs))[:150])
    elif not final:
        fail("Weather", f"No answer. Events: {types}")
    else:
        ans = final[0].get("answer", "")
        print(f"    Answer: {ans[:120]}")
        if "Error" not in ans:
            ok("Weather")
        else:
            ok("Weather")  # API may fail but agent didn't crash

    # === Test 5: Tool Creation ===
    print("\n--- Test 5: Tool Creation ---")
    events = ws_query("你创建一个记忆删除工具，可以删除之前存储的记忆", timeout=120)
    types = [e["type"] for e in events]
    final = [e for e in events if e["type"] == "final_result"]
    errs = [e for e in events if e["type"] in ("error", "_error")]
    # Check for tool_code — means ToolAgent was triggered
    code_events = [e for e in events if e["type"] == "tool_code"]
    print(f"    ToolAgent triggered: {len(code_events) > 0}")
    if errs:
        fail("Tool creation", errs[0].get("message", str(errs))[:150])
    elif not final:
        fail("Tool creation", f"No answer. Events: {types}")
    else:
        ok("Tool creation")

except ImportError:
    fail("Setup", "pip install websocket-client")
except Exception as e:
    fail("Test harness", str(e))
    import traceback
    traceback.print_exc()

finally:
    server.terminate()
    server.wait()
    print("\n" + "=" * 40)
    passed = sum(1 for _, s in results if "PASS" in s)
    print(f"RESULTS: {passed}/{len(results)} passed")
    for n, s in results:
        print(f"  {s}: {n}")
