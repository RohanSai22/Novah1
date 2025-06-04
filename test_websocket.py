#!/usr/bin/env python3
"""
Simple WebSocket test client to verify real-time updates are working
"""
import asyncio
import json
import websockets
import time

async def test_websocket():
    """Test WebSocket connection and real-time updates"""
    uri = "ws://localhost:8001/ws/updates"
    
    try:
        print("🔌 Connecting to WebSocket...")
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to WebSocket at", uri)
            print("🎧 Listening for real-time updates...")
            
            # Listen for updates for 30 seconds
            start_time = time.time()
            message_count = 0
            
            while time.time() - start_time < 30:
                try:
                    # Wait for message with timeout
                    message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    data = json.loads(message)
                    message_count += 1
                    
                    print(f"\n📨 Message #{message_count} received:")
                    print(f"   Type: {data.get('type', 'unknown')}")
                    print(f"   Status: {data.get('status', 'unknown')}")
                    print(f"   Agent: {data.get('current_agent', 'unknown')}")
                    print(f"   Step: {data.get('current_step', 0)}/{data.get('total_steps', 0)}")
                    
                    if data.get('type') == 'execution_complete':
                        print("🎉 Execution completed!")
                        break
                        
                except asyncio.TimeoutError:
                    print(".", end="", flush=True)
                    continue
                except websockets.exceptions.ConnectionClosed:
                    print("\n❌ WebSocket connection closed")
                    break
                    
            print(f"\n📊 Test completed: {message_count} messages received")
            
    except Exception as e:
        print(f"❌ WebSocket test failed: {e}")

if __name__ == "__main__":
    print("🧪 Starting WebSocket test...")
    asyncio.run(test_websocket())
