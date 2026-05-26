# DK-2500 开发套件规格摘要

## 核心硬件

| 项目 | 规格 |
|------|------|
| CPU | Intel Core Ultra 5 225U（基础功耗 15W，最高 4.8 GHz） |
| 内存 | 双通道 DDR5 SO-DIMM，2 × 8 GB |
| 存储 | 128 GB SSD（M.2 2280 M-Key） |
| BIOS | AMI 256 Mbit SPI Flash |

## 显示接口

- **DP**：1 × DP1.4a，支持 4096×2160 @ 60Hz（36bpp）
- **HDMI**：1 × HDMI2.1，支持 4096×2304 @ 60Hz
- **LCD**：1 × eDP 1.4b（4096×2304 @ 60Hz）或 LVDS（单通道 18/24-bit，最高 1920×1200）

## 网络与无线扩展

- **以太网**：4 × Intel I210 GbE LAN
- **无线模组接口**：
  - M.2 E-Key（2230）：Wi‑Fi / BT（可选 Wi‑Fi5 / Wi‑Fi6）
  - M.2 B-Key（3042/3052）：4G / 5G 蜂窝模组
  - NANO SIM 卡槽

## USB 接口

- 4 × USB 3.2：
  - 1 × USB Type‑C
  - 3 × USB Type‑A

## 40-Pin JTAG 调试接口（重点）

**电气特性**：引脚 1 为 3.3V，引脚 2/4 为 5V，引脚 17 为 1.8V，偶数引脚多为 GND。

### 引脚定义表

| 引脚 | 信号 | 引脚 | 信号 |
|------|------|------|------|
| 1 | 3.3V | 2 | 5V |
| 3 | I2C1_DAT / GPIO14 | 4 | 5V |
| 5 | I2C1_CLK / GPIO2 | 6 | GND |
| 7 | GPIO3 | 8 | UART_TX (SIO) |
| 9 | GND | 10 | UART_RX (SIO) |
| 11 | RTS (SIO) | 12 | I2S_BCLK / GPIO18 |
| 13 | GPIO5 | 14 | GND |
| 15 | GPIO6 | 16 | GPIO19 |
| 17 | 1.8V | 18 | GPIO20 |
| 19 | GSPI_MOSI | 20 | GND |
| 21 | GSPI_MISO | 22 | GPIO21 |
| 23 | GSPI_CLK | 24 | GSPI_CS0 / GPIO22 |
| 25 | GND | 26 | GPIO23 |
| 27 | I2C0_DAT / GPIO10 | 28 | I2C0_CLK / GPIO24 |
| 29 | GPIO11 | 30 | GND |
| 31 | GPIO12 | 32 | PWM0 / GPIO25 |
| 33 | GPIO13 | 34 | GND |
| 35 | I2S_SYNC / GPIO14 | 36 | CTS (SIO) |
| 37 | GPIO15 | 38 | I2S_SDI / GPIO27 |
| 39 | GND | 40 | I2S_SDO / GPIO28 |

> **注意**：多路复用引脚（如 I2C、PWM、GSPI、I2S）需通过寄存器配置功能选择。

## 其他扩展总线

- **SMBus**：1 路
- **CON1**：4 COM / 3 PCIe / 2 USB3.0 / 1 USB2.0
- **CON2**：4 PCIe / GPIO

## 电源与功耗

| 参数 | 值 |
|------|------|
| 输入电压 | 24V ±10% DC（3 位凤凰端子） |
| 最大功耗 | 60 W |
| 电源管理 | AT / ATX 可配置 |
| RTC | 板载电池接口 |

## 环境与物理参数

| 参数 | 范围 |
|------|------|
| 工作温度 | 0 ~ 60℃ |
| 存储温度 | -40 ~ 85℃ |
| 工作湿度 | 95% @ 40℃（无冷凝） |
| 尺寸 | 200 × 215 mm |

## 操作系统支持

- Windows 10 / 11
- Ubuntu 20.04 / 22.04
- CentOS 8

## 其他板载功能

- **TPM**：板载 TPM2.0（可选）
- **看门狗**：65536 级，0～65535 秒
- **LED**：硬盘灯、内存灯、2 × 用户自定义灯
- **按键/开关**：电源键、复位键、远程开关、Clear CMOS

## 订购与配件（摘要）

### 开发套件型号
- **DK2500-U001**：225U CPU，4×GbE，30‑pin JTAG，24V 输入

### 可选无线模组
| 型号 | 类型 |
|------|------|
| Wi‑Fi5/BT (9560.NGWG.NV) | M.2 2230 CNVi |
| Wi‑Fi6/BT (AX201.NGWG) | M.2 2230 CNVi |
| 5G (RM520N / RM500U) | M.2 B‑Key |
| 4G (EM05CEFC) | M.2 B‑Key（仅中国） |

### 包装清单
- DK-2500 开发板 ×1
- 电源适配器 ×1
- 圆头螺丝 ×4
- 三插电源线 ×1