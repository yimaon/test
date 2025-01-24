import socket
import cv2
import numpy as np
import struct
import io
import logging
import time
from typing import Optional, Tuple
import queue
import zlib
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import tempfile

class VideoStreamClient:
    def __init__(self, host: str = '127.0.0.1', port: int = 8027,
                 window_name: str = 'Video Stream', 
                 window_size: Optional[Tuple[int, int]] = None):
        # Network settings
        self.host = host
        self.port = port
        self.buffer_size = 65536
        self.packet_size = 4096
        self.max_frame_size = 10_000_000  # 10MB
        self.socket_timeout = 5.0
        
        # Connection parameters
        self.max_retries = 3
        self.retry_delay = 2.0
        self.frame_timeout = 1.0
        self.connect_timeout = 10.0
        self.reconnect_timeout = 3.0
        self.max_reconnects = 5
        self.handshake_timeout = 2.0
        self.receive_timeout = 0.5
        
        # Protocol settings
        self.handshake_magic = b'CCTV'
        self.protocol_version = 1
        self.frame_sync_marker = b'\xFF\xD8\xFF'
        self.frame_end_marker = b'\xFF\xD9'
        self.frame_header_size = 8
        
        # Window settings
        self.window_name = window_name
        self.window_size = window_size or (800, 600)
        
        # Image enhancement settings
        self.image_enhance = True
        self.denoise_strength = 2
        self.sharpen_strength = 1.0
        self.brightness_alpha = 1.0
        self.brightness_beta = 0
        self.saturation_scale = 1.0
        self.gamma = 1.0
        self.detail_preserve = True
        self.color_balance = False
        self.maintain_original = True
        
        # State variables
        self.running = False
        self.client: Optional[socket.socket] = None
        self.frame_buffer = bytearray()
        self.sync_attempts = 0
        self.max_sync_attempts = 10
        self.reconnect_count = 0
        self.last_connection_attempt = 0
        
        # Connection retry settings
        self.min_reconnect_delay = 1.0  # Add missing attribute
        self.current_reconnect_delay = self.min_reconnect_delay
        self.connection_backoff = 1.5
        self.max_reconnect_delay = 30.0
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # H.264 settings
        self.codec = cv2.VideoWriter_fourcc(*'H264')
        self.bitrate = 4_000_000  # 4 Mbps
        self.gop_size = 30
        self.color_format = 'bgr24'
        
        # Color correction settings
        self.color_profile = {
            'temperature': 6500,  # Kelvin
            'tint': 0,           # Green-Magenta balance
            'matrix': 'srgb'     # Color space matrix
        }
        
        # Advanced image processing
        self.color_range = 'full'  # or 'studio'
        self.color_primaries = 'bt709'
        self.color_depth = 8
        self.yuv_format = cv2.COLOR_YUV2BGR_NV12

        # Add color matrix definitions
        self.srgb_matrix = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ], dtype=np.float32)
        
        # เพิ่มตัวแปรควบคุมการรักษาสีต้นฉบับ
        self.preserve_original_colors = True  # เพิ่มบรรทัดนี้
        
    def setup_window(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, *self.window_size)
        
    def connect(self) -> bool:
        current_time = time.time()
        
        # ตรวจสอบว่าผ่านเวลา delay ที่กำหนดหรือยัง
        if current_time - self.last_connection_attempt < self.current_reconnect_delay:
            time.sleep(0.1)  # รอสักครู่ก่อนลองใหม่
            return False

        self.last_connection_attempt = current_time

        try:
            if self.client:
                try:
                    self.client.shutdown(socket.SHUT_RDWR)
                except:
                    pass
                self.client.close()
                time.sleep(0.5)

            self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 131072)
            self.client.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            self.client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            
            # เพิ่มการตั้งค่า TCP keepalive
            self.client.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            self.client.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 30)
            self.client.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10)
            self.client.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3)

            self.client.settimeout(self.connect_timeout)
            self.client.connect((self.host, self.port))
            self.client.settimeout(self.receive_timeout)
            
            # Handshake
            if not self.perform_handshake():
                raise Exception("Handshake failed")
                
            self.client.settimeout(self.frame_timeout)
            
            # รีเซ็ต reconnect delay เมื่อเชื่อมต่อสำเร็จ
            self.current_reconnect_delay = self.min_reconnect_delay
            logging.info(f"Connected to {self.host}:{self.port}")
            return True

        except Exception as e:
            if self.client:
                self.client.close()
                self.client = None
                
            # เพิ่ม reconnect delay แบบ exponential backoff
            self.current_reconnect_delay = min(
                self.current_reconnect_delay * self.connection_backoff,
                self.max_reconnect_delay
            )
            
            logging.error(f"Connection failed: {e}, next retry in {self.current_reconnect_delay:.1f}s")
            return False
            
    def perform_handshake(self) -> bool:
        try:
            # 1. ส่ง magic number และ protocol version
            handshake_data = struct.pack('!4sI', self.handshake_magic, self.protocol_version)
            self.client.settimeout(self.handshake_timeout)
            self.client.sendall(handshake_data)
            
            # 2. รับการตอบกลับ
            response = self._receive_exactly(5)  # 1 byte status + 4 bytes protocol version
            if not response:
                logging.error("No handshake response")
                return False
                
            status = response[0]
            if status != 0x01:  # 0x01 = OK
                logging.error(f"Handshake rejected with status: {status}")
                return False
                
            server_version = struct.unpack('!I', response[1:])[0]
            if server_version != self.protocol_version:
                logging.error(f"Protocol version mismatch: client={self.protocol_version}, server={server_version}")
                return False

            # 3. ส่งขนาดหน้าต่าง
            window_config = struct.pack('!II', *self.window_size)
            self.client.sendall(window_config)
            
            # 4. รอรับการยืนยันการตั้งค่า
            config_ack = self._receive_exactly(1)
            if not config_ack or config_ack[0] != 0x01:
                logging.error("Window config rejected")
                return False

            logging.info("Handshake completed successfully")
            return True

        except socket.timeout:
            logging.error("Handshake timeout")
            return False
        except Exception as e:
            logging.error(f"Handshake error: {e}")
            return False
            
    def receive_frame_size(self) -> Optional[int]:
        try:
            size_data = self.client.recv(4)
            if not size_data or len(size_data) != 4:
                return None
            return struct.unpack('!I', size_data)[0]
        except socket.timeout:
            return None
        except Exception as e:
            logging.error(f"Error receiving frame size: {e}")
            return None

    def receive_frame_data(self) -> Optional[bytes]:
        try:
            # 1. Clear old data from frame buffer if too large
            if len(self.frame_buffer) > 10*1024*1024:  # 10MB limit
                self.frame_buffer.clear()

            # 2. Read data until we find sync marker
            while True:
                if self.sync_attempts >= self.max_sync_attempts:
                    self.sync_attempts = 0
                    self.frame_buffer.clear()
                    return None

                # Look for frame sync marker in buffer
                if len(self.frame_buffer) >= 3:
                    sync_idx = self.frame_buffer.find(self.frame_sync_marker)
                    if sync_idx >= 0:
                        # Remove data before sync marker
                        if (sync_idx > 0):
                            self.frame_buffer = self.frame_buffer[sync_idx:]
                        break
                
                # Read more data
                try:
                    chunk = self.client.recv(self.buffer_size)
                    if not chunk:
                        return None
                    self.frame_buffer.extend(chunk)
                except socket.timeout:
                    self.sync_attempts += 1
                    continue
                except Exception as e:
                    logging.error(f"Error reading data: {e}")
                    return None

            # 3. Look for frame end marker
            while True:
                end_idx = self.frame_buffer.find(self.frame_end_marker)
                if end_idx >= 0:
                    # Found complete frame
                    frame_data = self.frame_buffer[:end_idx + 2]
                    self.frame_buffer = self.frame_buffer[end_idx + 2:]
                    self.sync_attempts = 0
                    
                    # Validate frame
                    if self.validate_jpeg(frame_data):
                        return bytes(frame_data)
                    return None

                # Read more data
                try:
                    chunk = self.client.recv(self.buffer_size)
                    if not chunk:
                        return None
                    self.frame_buffer.extend(chunk)
                except socket.timeout:
                    self.sync_attempts += 1
                    if self.sync_attempts >= self.max_sync_attempts:
                        self.sync_attempts = 0
                        self.frame_buffer.clear()
                        return None
                    continue
                except Exception as e:
                    logging.error(f"Error reading frame data: {e}")
                    return None

        except Exception as e:
            logging.error(f"Frame receive error: {e}")
            return None

    def validate_jpeg(self, data: bytes) -> bool:
        """Validate JPEG data"""
        try:
            # Check JPEG markers
            if not data.startswith(b'\xFF\xD8') or not data.endswith(b'\xFF\xD9'):
                return False
                
            # Quick validation by trying to decode
            img_array = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return img is not None
            
        except Exception:
            return False

    def _receive_exactly(self, size: int, retry: bool = True) -> Optional[bytes]:
        """Helper to receive exact number of bytes with improved error handling"""
        data = bytearray()
        retries = 3 if retry else 1
        start_time = time.time()
        
        for attempt in range(retries):
            try:
                while len(data) < size:
                    if time.time() - start_time > self.frame_timeout:
                        if attempt == retries - 1:  # Last attempt
                            return None
                        break  # Try next attempt
                        
                    remaining = size - len(data)
                    chunk = self.client.recv(min(remaining, self.buffer_size))
                    if not chunk:
                        if attempt == retries - 1:
                            return None
                        break
                        
                    data.extend(chunk)
                    
                if len(data) == size:
                    return bytes(data)
                    
            except socket.timeout:
                if not retry or attempt == retries - 1:
                    return None
            except Exception as e:
                logging.error(f"Receive error: {e}")
                return None
                
        return None
            
    def process_frame(self, frame_data: bytes) -> Optional[np.ndarray]:
        try:
            # H.264 decode หรือ JPEG decode
            if frame_data.startswith(b'\x00\x00\x00\x01'):
                frame_array = np.frombuffer(frame_data, dtype=np.uint8)
                frame = self._decode_h264(frame_array)
            else:
                frame_array = np.frombuffer(frame_data, dtype=np.uint8)
                frame = cv2.imdecode(frame_array, cv2.IMREAD_UNCHANGED)  # ใช้ UNCHANGED แทน COLOR

            if frame is None:
                return None

            if self.preserve_original_colors:
                # ถ้าต้องการรักษาสีต้นฉบับ ไม่ต้องแปลง color space
                return frame
            else:
                # กรณีต้องการปรับแต่งสี
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = self._apply_color_correction(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                return frame

        except Exception as e:
            logging.error(f"Frame processing error: {e}")
            return None

    def _decode_h264(self, frame_data: np.ndarray) -> Optional[np.ndarray]:
        """Decode H.264 frame with proper color handling"""
        try:
            # Create VideoCapture from memory buffer
            with tempfile.NamedTemporaryFile(suffix='.h264') as tmp:
                tmp.write(frame_data.tobytes())
                tmp.flush()
                cap = cv2.VideoCapture(tmp.name)
                ret, frame = cap.read()
                cap.release()
                
            if not ret:
                return None
                
            # Convert from YUV to RGB with proper color range
            if self.color_range == 'studio':
                frame = cv2.normalize(frame, None, 16, 235, cv2.NORM_MINMAX)
            
            return frame
            
        except Exception as e:
            logging.error(f"H.264 decode error: {e}")
            return None

    def _apply_color_correction(self, frame: np.ndarray) -> np.ndarray:
        """Apply professional color correction (ทำงานกับ RGB color space)"""
        try:
            if not self.image_enhance:
                return frame

            # Convert to float32 and normalize
            frame = frame.astype(np.float32) / 255.0
            
            # White balance correction
            if self.color_balance:
                frame = self._apply_white_balance(frame)
            
            # Color matrix transformation using RGB color space
            if self.color_profile['matrix'] == 'srgb':
                frame = self._apply_color_matrix(frame, self.srgb_matrix)
            
            # Color temperature adjustment
            frame = self._adjust_color_temperature(frame)
            
            # Convert back to uint8
            frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
            
            return frame
            
        except Exception as e:
            logging.error(f"Color correction error: {e}")
            if isinstance(frame, np.ndarray):
                return frame.astype(np.uint8)
            return frame

    def _apply_white_balance(self, frame: np.ndarray) -> np.ndarray:
        """Auto white balance with gray world assumption"""
        try:
            # Convert to float32 for calculations
            frame = frame.astype(np.float32)
            b, g, r = cv2.split(frame)
            b_avg, g_avg, r_avg = np.mean(b), np.mean(g), np.mean(r)
            
            # Calculate gains
            gray_target = (b_avg + g_avg + r_avg) / 3
            b_gain = gray_target / b_avg if b_avg > 0 else 1
            g_gain = gray_target / g_avg if g_avg > 0 else 1
            r_gain = gray_target / r_avg if r_avg > 0 else 1
            
            # Convert gains to float32 arrays
            b_gain = np.full_like(b, b_gain, dtype=np.float32)
            g_gain = np.full_like(g, g_gain, dtype=np.float32)
            r_gain = np.full_like(r, r_gain, dtype=np.float32)
            
            # Apply gains
            b = cv2.multiply(b, b_gain)
            g = cv2.multiply(g, g_gain)
            r = cv2.multiply(r, r_gain)
            
            # Merge channels back
            return cv2.merge([b, g, r])
            
        except Exception as e:
            logging.error(f"White balance error: {e}")
            return frame

    def _apply_color_matrix(self, frame: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Apply color transformation matrix"""
        try:
            # Reshape frame for matrix multiplication
            height, width = frame.shape[:2]
            pixels = frame.reshape(-1, 3)
            
            # Apply color matrix
            pixels = np.dot(pixels, matrix.T)
            
            # Clip values and reshape back
            pixels = np.clip(pixels, 0, 1)
            frame = pixels.reshape(height, width, 3)
            
            return frame
            
        except Exception as e:
            logging.error(f"Color matrix error: {e}")
            return frame

    def _adjust_color_temperature(self, frame: np.ndarray) -> np.ndarray:
        """Adjust color temperature with improved algorithm"""
        try:
            temp = self.color_profile['temperature']
            # Calculate RGB multipliers based on temperature
            if temp <= 6500:
                # Warm (yellow-red)
                red = 1.0
                blue = 0.8 + 0.2 * (temp / 6500.0)
                green = 0.9 + 0.1 * (temp / 6500.0)
            else:
                # Cool (blue)
                factor = (temp - 6500) / 3500.0  # Up to 10000K
                red = 1.0 - 0.1 * factor
                blue = 1.0 + 0.1 * factor
                green = 1.0
            
            # Convert to float32 arrays matching frame shape
            b, g, r = cv2.split(frame)
            blue_gain = np.full_like(b, blue, dtype=np.float32)
            green_gain = np.full_like(g, green, dtype=np.float32)
            red_gain = np.full_like(r, red, dtype=np.float32)
            
            # Apply color adjustment
            b = cv2.multiply(b, blue_gain)
            g = cv2.multiply(g, green_gain)
            r = cv2.multiply(r, red_gain)
            
            return cv2.merge([b, g, r])
            
        except Exception as e:
            logging.error(f"Color temperature adjustment error: {e}")
            return frame

    def _get_srgb_matrix(self) -> np.ndarray:
        """Get sRGB color transformation matrix"""
        return self.srgb_matrix

    def toggle_image_enhancement(self):
        """เปิด/ปิดการปรับแต่งภาพ"""
        self.image_enhance = not self.image_enhance

    def toggle_color_preservation(self):
        """สลับโหมดระหว่างการรักษาสีต้นฉบับกับการปรับแต่งสี"""
        self.preserve_original_colors = not self.preserve_original_colors
        logging.info(f"Color preservation: {'enabled' if self.preserve_original_colors else 'disabled'}")
            
    def run(self):
        consecutive_failures = 0
        max_failures = 5  # ลดจำนวน failures ที่ยอมรับได้
        retry_delay = 2.0

        while self.reconnect_count < self.max_reconnects:
            try:
                if not self.connect():
                    self.reconnect_count += 1
                    time.sleep(self.retry_delay)
                    continue

                self.reconnect_count = 0
                self.running = True
                self.setup_window()
                self.frame_buffer.clear()
                self.sync_attempts = 0
                consecutive_failures = 0

                while self.running:
                    frame_data = self.receive_frame_data()
                    if frame_data is None:
                        consecutive_failures += 1
                        if consecutive_failures >= self.max_timeouts:
                            logging.warning(f"Too many consecutive failures ({consecutive_failures})")
                            break
                        time.sleep(0.1)
                        continue

                    consecutive_failures = 0
                    frame = self.process_frame(frame_data)
                    
                    if frame is not None:
                        cv2.imshow(self.window_name, frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            self.running = False
                            break
                        elif key == ord('o'):  # เพิ่มปุ่ม 'o' สำหรับสลับโหมดสี
                            self.toggle_color_preservation()

            except ConnectionError:
                logging.error("Connection lost")
                time.sleep(self.retry_delay)
            except Exception as e:
                logging.error(f"Runtime error: {e}")
                time.sleep(self.retry_delay)

            if not self.running:
                break

        self.cleanup()

    def toggle_fullscreen(self):
        current_state = cv2.getWindowProperty(
            self.window_name, 
            cv2.WND_PROP_FULLSCREEN
        )
        cv2.setWindowProperty(
            self.window_name,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN if current_state != cv2.WINDOW_FULLSCREEN else cv2.WINDOW_NORMAL
        )
            
    def cleanup(self):
        self.running = False
        if self.client:
            self.client.close()
        cv2.destroyAllWindows()
        logging.info("Client cleaned up")

def main():
    max_restarts = 5
    restart_delay = 3
    restart_count = 0
    
    while restart_count < max_restarts:
        try:
            client = VideoStreamClient(
                host='100.99.34.117', 
                port=8027,
                window_name='CCTV Stream',
                window_size=(1280, 720)
            )
            client.run()
            
            # ถ้าออกจาก run() ด้วยการกด 'q'
            if not client.running:
                break
                
            restart_count += 1
            logging.info(f"Restarting client (attempt {restart_count}/{max_restarts})")
            time.sleep(restart_delay)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            logging.error(f"Critical error: {e}")
            restart_count += 1
            time.sleep(restart_delay)

    logging.info("Client terminated")

if __name__ == '__main__':
    main()

class VideoClient:
    def __init__(self, host: str = '100.99.34.117', port: int = 8027):
        # Core settings
        self.host = host
        self.port = port
        self.socket = None
        self.running = False
        
        # Connection settings
        self.connect_timeout = 10.0
        self.reconnect_delay = 2.0
        self.max_retries = 3
        
        # Video settings
        self.resolution = (1920, 1080)
        self.buffer_size = 65536
        self.compression_level = 1
        self.max_frame_size = 20_000_000
        self.frame_timeout = 5.0
        self.jpeg_quality = 95
        self.frame_queue_size = 10
        
        # Image enhancement settings
        self.image_enhance = True
        self.denoise_strength = 5
        self.sharpen_strength = 1.2
        self.brightness_alpha = 1.1
        self.brightness_beta = 10
        self.gamma = 1.2
        self.auto_enhance = True
        
        # Initialize display window with error handling
        try:
            cv2.namedWindow('Video Stream', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Video Stream', *self.resolution)
        except Exception as e:
            print(f"Warning: Could not initialize window: {e}")
            
    def connect(self) -> bool:
        try:
            if self.socket:
                self.socket.close()
            
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.connect_timeout)
            self.socket.connect((self.host, self.port))
            self.socket.settimeout(self.frame_timeout)
            return True
        except Exception as e:
            print(f"Connection error: {e}")
            return False

    def process_frame(self, frame):
        try:
            if frame is None:
                return None
                
            if not self.image_enhance:
                return cv2.resize(frame, self.resolution, interpolation=cv2.INTER_LANCZOS4)

            # Auto brightness adjustment
            if self.auto_enhance:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                mean_brightness = np.mean(gray)
                if (mean_brightness < 80):
                    self.brightness_alpha = 1.2
                    self.brightness_beta = 15
                elif (mean_brightness > 200):
                    self.brightness_alpha = 0.8
                    self.brightness_beta = -10
                else:
                    self.brightness_alpha = 1.1
                    self.brightness_beta = 5

            # Enhanced processing pipeline
            frame = cv2.resize(frame, self.resolution, interpolation=cv2.INTER_LANCZOS4)
            frame = cv2.fastNlMeansDenoisingColored(frame, None, 
                                                   self.denoise_strength, 
                                                   self.denoise_strength, 7, 21)
            
            kernel = np.array([[-1,-1,-1],
                             [-1, 9,-1],
                             [-1,-1,-1]]) * self.sharpen_strength / 9
            frame = cv2.filter2D(frame, -1, kernel)
            
            frame = cv2.convertScaleAbs(frame, 
                                      alpha=self.brightness_alpha, 
                                      beta=self.brightness_beta)

            if self.gamma != 1.0:
                inv_gamma = 1.0 / self.gamma
                table = np.array([((i / 255.0) ** inv_gamma) * 255
                                for i in np.arange(0, 256)]).astype("uint8")
                frame = cv2.LUT(frame, table)

            return frame
            
        except Exception as e:
            print(f"Frame processing error: {e}")
            return None

    def receive_frame(self):
        try:
            # รับขนาดเฟรม
            size_data = self.socket.recv(8)
            if not size_data:
                return None
                
            message_size = struct.unpack("Q", size_data)[0]
            if message_size > self.max_frame_size:
                raise ValueError("Frame too large")

            # รับข้อมูลเฟรมแบบ chunks
            frame_data = bytearray()
            remaining = message_size
            
            while remaining > 0:
                chunk_size = min(remaining, self.buffer_size)
                chunk = self.socket.recv(chunk_size)
                if not chunk:
                    return None
                frame_data.extend(chunk)
                remaining -= len(chunk)

            # Decompress และแปลงเป็นเฟรม
            decompressed = zlib.decompress(frame_data)
            frame_array = np.frombuffer(decompressed, dtype=np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            
            return self.process_frame(frame)
            
        except Exception as e:
            print(f"Frame receive error: {e}")
            return None

    def run(self):
        retry_count = 0
        self.running = True
        
        while self.running and retry_count < self.max_retries:
            if not self.connect():
                retry_count += 1
                print(f"Connection attempt {retry_count}/{self.max_retries} failed")
                time.sleep(self.reconnect_delay)
                continue
                
            retry_count = 0
            try:
                while self.running:
                    frame = self.receive_frame()
                    if frame is not None:
                        cv2.imshow("Video Stream", frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            self.running = False
                        elif key == ord('f'):
                            self.toggle_fullscreen()
                        elif key == ord('e'):
                            self.image_enhance = not self.image_enhance
                    else:
                        break
            except Exception as e:
                print(f"Stream error: {e}")
                time.sleep(1)

        self.cleanup()

    def toggle_fullscreen(self):
        try:
            current_state = cv2.getWindowProperty('Video Stream', 
                                                cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty('Video Stream', 
                                cv2.WND_PROP_FULLSCREEN,
                                cv2.WINDOW_FULLSCREEN 
                                if current_state != cv2.WINDOW_FULLSCREEN 
                                else cv2.WINDOW_NORMAL)
        except Exception as e:
            print(f"Failed to toggle fullscreen: {e}")

    def cleanup(self):
        self.running = False
        if self.socket:
            try:
                self.socket.shutdown(socket.SHUT_RDWR)
            except:
                pass
            self.socket.close()
        cv2.destroyAllWindows()
        print("Client cleaned up")

class VideoStreamGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("CCTV Video Stream")
        self.client = VideoStreamClient(
            host='100.99.34.117', 
            port=8027,
            window_name='CCTV Stream',
            window_size=(1280, 720)
        )
        
        # GUI Layout
        self.setup_gui()
        
        # Color correction settings
        self.color_temp = tk.DoubleVar(value=6500)  # Default 6500K
        self.saturation = tk.DoubleVar(value=1.0)
        self.contrast = tk.DoubleVar(value=1.0)
        self.brightness = tk.DoubleVar(value=0)
        
        # Auto adjustment settings
        self.auto_wb = tk.BooleanVar(value=True)
        self.auto_exposure = tk.BooleanVar(value=True)
        
        self.start_stream()

    def setup_gui(self):
        # Main container
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Video frame
        self.video_frame = ttk.Frame(self.main_frame)
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Control panel
        self.control_frame = ttk.Frame(self.main_frame, padding="5")
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Image controls
        ttk.Label(self.control_frame, text="Color Temperature (K)").pack()
        temp_scale = ttk.Scale(self.control_frame, from_=2000, to=12000,
                             variable=self.color_temp, orient=tk.HORIZONTAL)
        temp_scale.pack(fill=tk.X)
        
        ttk.Label(self.control_frame, text="Saturation").pack()
        sat_scale = ttk.Scale(self.control_frame, from_=0, to=2,
                            variable=self.saturation, orient=tk.HORIZONTAL)
        sat_scale.pack(fill=tk.X)
        
        # Auto adjustment toggles
        ttk.Checkbutton(self.control_frame, text="Auto White Balance",
                       variable=self.auto_wb).pack()
        ttk.Checkbutton(self.control_frame, text="Auto Exposure",
                       variable=self.auto_exposure).pack()

    def update_frame(self, frame):
        if frame is None:
            return
            
        # Color correction
        frame = self.apply_color_correction(frame)
        
        # Convert to PIL format
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        photo = ImageTk.PhotoImage(image=image)
        
        # Update display
        self.video_label.configure(image=photo)
        self.video_label.image = photo
        
        # Schedule next update
        self.after(10, self.get_next_frame)

    def apply_color_correction(self, frame):
        # Auto white balance
        if self.auto_wb.get():
            frame = self.auto_white_balance(frame)
        
        # Color temperature adjustment
        frame = self.adjust_color_temperature(frame, self.color_temp.get())
        
        # Saturation
        if self.saturation.get() != 1.0:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(float)
            hsv[:,:,1] = hsv[:,:,1] * self.saturation.get()
            hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)
            frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        return frame

    def auto_white_balance(self, frame):
        result = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

    def adjust_color_temperature(self, frame, temp):
        # Convert temperature to RGB adjustment
        temp = temp / 100
        
        if temp <= 66:
            red = 255
            green = 99.4708025861 * np.log(temp) - 161.1195681661
            blue = 138.5177312231 * np.log(temp - 10) - 305.0447927307
        else:
            red = 329.698727446 * np.power(temp - 60, -0.1332047592)
            green = 288.1221695283 * np.power(temp - 60, -0.0755148492)
            blue = 255
            
        # Normalize and apply
        rgb_scaling = np.array([
            min(255, max(0, red)) / 255,
            min(255, max(0, green)) / 255,
            min(255, max(0, blue)) / 255
        ])
        
        return cv2.multiply(frame, rgb_scaling)

    def get_next_frame(self):
        frame_data = self.client.receive_frame_data()
        if frame_data is not None:
            frame = self.client.process_frame(frame_data)
            self.update_frame(frame)

    def start_stream(self):
        if self.client.connect():
            self.get_next_frame()
        else:
            self.after(1000, self.start_stream)

    def cleanup(self):
        self.client.cleanup()
        self.quit()

def main():
    app = VideoStreamGUI()
    app.protocol("WM_DELETE_WINDOW", app.cleanup)
    app.mainloop()

if __name__ == '__main__':
    main()
