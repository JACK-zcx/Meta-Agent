"""Test basic conversation scenarios through the full agent pipeline."""
import sys, os, threading, time, json, subprocess

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')

results = []


def test(name, fn):
    try:
        fn()
        results.append((name, "PASS"))
        print(f"\n✅ {name}: PASS")
    except Exception as e:
        results.append((name, f"FAIL - {e}"))
        print(f"\n❌ {name}: FAIL - {e}")


# ── Start server ──
print("Starting server...")
python_exe = r"C:\Users\张崇轩\AppData\Local\Programs\Python\Python312\python.exe"
server_proc = subprocess.Popen(
    [python_exe, "server.py"],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    cwd=os.path.dirname(os.path.abspath(__file__)),
)
time.sleep(8)  # Wait for server to be ready

try:
    import websocket

    def send_and_wait(query, timeout=45):
        """Send query via WebSocket and collect all events until 'done'."""
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
            pass
        except Exception as e:
            events.append({"type": "_error", "message": str(e)})
        finally:
            ws.close()
        return events

    # ── Test 1: Simple greeting ──
    def test_greeting():
        events = send_and_wait("你好")
        types = [e["type"] for e in events]
        print(f"  Event types: {types}")
        # Should contain final_result or error, not crash
        final = [e for e in events if e["type"] == "final_result"]
        assert len(final) > 0, "No final_result event!"
        assert len(final[0].get("answer", "")) > 0, "Empty answer!"

    test("Simple greeting", test_greeting)

    # ── Test 2: Memory store ──
    def test_memory_store():
        events = send_and_wait("帮我记住：Python是世界上最受欢迎的编程语言之一")
        types = [e["type"] for e in events]
        print(f"  Event types: {types}")
        # Should NOT have 'error' events
        error_events = [e for e in events if e["type"] == "error"]
        assert len(error_events) == 0, f"Got error: {error_events}"
        final = [e for e in events if e["type"] == "final_result"]
        assert len(final) > 0, "No final_result event!"

    test("Memory store", test_memory_store)

    # ── Test 3: Memory recall ──
    def test_memory_recall():
        events = send_and_wait("我之前让你记住什么了？")
        final = [e for e in events if e["type"] == "final_result"]
        assert len(final) > 0, "No final_result event!"
        answer = final[0].get("answer", "")
        print(f"  Answer: {answer[:200]}...")
        assert "Python" in answer, f"Memory recall didn't find Python: {answer[:300]}"

    test("Memory recall", test_memory_recall)

    # ── Test 4: Tool creation ──
    def test_tool_creation():
        events = send_and_wait("你创建一个记忆删除工具，可以删除之前存储的记忆", timeout=120)
        types = [e["type"] for e in events]
        print(f"  Event types: {types}")

        # Check for tool_code event (indicates ToolAgent was triggered)
        code_events = [e for e in events if e["type"] == "tool_code"]
        print(f"  Tool code events: {len(code_events)}")
        if len(code_events) > 0:
            print(f"  Generated code preview: {code_events[0].get('code','')[:100]}")

        # Should NOT crash with error
        error_events = [e for e in events if e["type"] == "error"]
        if error_events:
            print(f"  Errors: {error_events}")
        # At minimum it should complete (have final_result or done)
        final = [e for e in events if e["type"] == "final_result"]
        assert len(final) > 0, f"No final_result! Events: {[e['type'] for e in events]}"

    test("Tool creation", test_tool_creation)

    # ── Test 5: Weather query ──
    def test_weather():
        events = send_and_wait("北京今天天气怎么样？", timeout=120)
        error_events = [e for e in events if e["type"] == "error"]
        if error_events:
            print(f"  Errors: {error_events}")
        final = [e for e in events if e["type"] == "final_result"]
        assert len(final) > 0, "No final_result event!"
        print(f"  Answer: {final[0].get('answer','')[:200]}...")

    test("Weather query", test_weather)

    # ── Test 6: RAG document ──
    def test_rag():
        # Add document via WebSocket JSON message
        ws = websocket.create_connection("ws://localhost:8000/ws", timeout=5)
        ws.send(json.dumps({
            "type": "rag_add",
            "title": "AI Research",
            "text": "Artificial Intelligence (AI) is intelligence demonstrated by machines. Machine learning is a subset of AI that enables systems to learn from data. Deep learning uses neural networks with many layers."
        }))
        time.sleep(2)
        # Drain events
        events = []
        try:
            while True:
                msg = ws.recv()
                if not msg:
                    break
                events.append(json.loads(msg))
                if events[-1].get("type") == "rag_confirm":
                    break
        except:
            pass
        ws.close()

        confirm = [e for e in events if e["type"] == "rag_confirm"]
        assert len(confirm) > 0, f"No rag_confirm! Events: {[e['type'] for e in events]}"
        print(f"  RAG: {confirm[0].get('message','')}")

        # Now query about the doc
        events2 = send_and_wait("AI的机器学习子集是什么？")
        final = [e for e in events2 if e["type"] == "final_result"]
        assert len(final) > 0, "No final_result!"

    test("RAG document ingestion", test_rag)

except ImportError:
    print("\n⚠️ websocket-client not installed. Install with: pip install websocket-client")
    results.append(("All tests", "SKIPPED - missing websocket-client"))

finally:
    print("\n=== SUMMARY ===")
    for name, status in results:
        print(f"  {status}: {name}")
    server_proc.terminate()
    server_proc.wait()
    print("Server stopped.")

# Clean up
try:
    os.remove("test_conversation.py")
except:
    pass
