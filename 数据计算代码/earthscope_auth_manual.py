"""
EarthScope OAuth2认证脚本（改进版 - 手动处理流程）
"""

from pathlib import Path
import json
import time

try:
    from earthscope_sdk.auth.device_code_flow import DeviceCodeFlow
    from earthscope_sdk.common.context import SdkContext
    import httpx
except ImportError:
    print("请先安装: pip install earthscope-sdk")
    exit(1)

def authenticate_earthscope_manual():
    """手动处理认证流程"""
    print("=" * 60)
    print("EarthScope OAuth2 认证")
    print("=" * 60)

    # Token保存路径
    token_path = Path("download_lables/access_token_new.txt")
    token_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nToken保存路径: {token_path}")
    print("账号: wzl091002@gmail.com")
    print("-" * 60)

    # 检查现有token
    if token_path.exists():
        try:
            with open(token_path, 'r') as f:
                token_data = json.load(f)

            import datetime
            expires_at = token_data.get('expires_at', 0)

            if datetime.datetime.now().timestamp() < expires_at:
                print("\n[成功] 找到有效token")
                expire_time = datetime.datetime.fromtimestamp(expires_at)
                print(f"过期时间: {expire_time}")
                print(f"Token (前50字符): {token_data['access_token'][:50]}...")
                return True
            else:
                print("\n[提示] Token已过期，开始重新认证")
        except:
            print("\n[提示] 开始认证")

    # EarthScope OAuth2端点
    auth0_domain = "login.earthscope.org"
    client_id = "b9DtAFBd6QvMg761vI3YhYquNZbJX5G0"  # EarthScope公开client_id
    audience = "https://account.earthscope.org"

    print("\n" + "=" * 60)
    print("开始OAuth2设备代码流")
    print("=" * 60)

    try:
        # 步骤1: 请求设备代码
        print("\n[步骤1] 请求设备代码...")

        device_code_url = f"https://{auth0_domain}/oauth/device/code"
        device_code_data = {
            "client_id": client_id,
            "scope": "openid profile email offline_access",
            "audience": audience
        }

        with httpx.Client() as client:
            response = client.post(device_code_url, data=device_code_data, timeout=30)

            if response.status_code == 200:
                device_data = response.json()

                device_code = device_data.get('device_code')
                user_code = device_data.get('user_code')
                verification_uri = device_data.get('verification_uri_complete') or device_data.get('verification_uri')
                expires_in = device_data.get('expires_in', 900)
                interval = device_data.get('interval', 5)

                print(f"[成功] 获取到设备代码")
                print()
                print("=" * 60)
                print("请在浏览器中完成以下操作：")
                print("=" * 60)
                print(f"\n1. 在浏览器中打开: {verification_uri}")
                print(f"\n2. 输入设备代码: {user_code}")
                print(f"\n3. 使用以下账号登录:")
                print(f"   邮箱: wzl091002@gmail.com")
                print(f"   密码: wzl@8361241")
                print(f"\n4. 点击授权按钮")
                print(f"\n5. 返回此窗口等待完成")
                print()
                print("=" * 60)
                print(f"设备代码有效期: {expires_in//60} 分钟")
                print("等待授权中... (按Ctrl+C取消)")
                print("=" * 60)
                print()

                # 步骤2: 轮询等待授权
                token_url = f"https://{auth0_domain}/oauth/token"
                token_data = {
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    "device_code": device_code,
                    "client_id": client_id
                }

                max_attempts = expires_in // interval
                for attempt in range(max_attempts):
                    time.sleep(interval)

                    print(f"轮询中... ({attempt+1}/{max_attempts})", end='\r')

                    token_response = client.post(token_url, data=token_data, timeout=30)

                    if token_response.status_code == 200:
                        token_result = token_response.json()

                        access_token = token_result.get('access_token')
                        refresh_token = token_result.get('refresh_token')
                        expires_in_token = token_result.get('expires_in', 28800)

                        if access_token:
                            print("\n\n[成功] 认证成功！")

                            # 保存token
                            import datetime
                            token_save_data = {
                                'access_token': access_token,
                                'refresh_token': refresh_token,
                                'expires_at': int(datetime.datetime.now().timestamp() + expires_in_token),
                                'issued_at': int(datetime.datetime.now().timestamp()),
                                'scope': token_result.get('scope', '')
                            }

                            with open(token_path, 'w') as f:
                                json.dump(token_save_data, f, indent=2)

                            print(f"\nToken已保存到: {token_path}")
                            expire_time = datetime.datetime.fromtimestamp(token_save_data['expires_at'])
                            print(f"过期时间: {expire_time}")
                            print(f"有效期: {expires_in_token//3600} 小时")
                            print(f"Token (前50字符): {access_token[:50]}...")

                            print("\n" + "=" * 60)
                            print("认证完成！现在可以下载数据了")
                            print("=" * 60)
                            print("运行测试: python test_earthscope.py")

                            return True

                    elif token_response.status_code == 400:
                        error_data = token_response.json()
                        error = error_data.get('error', '')

                        if error == 'authorization_pending':
                            # 继续等待
                            continue
                        elif error == 'slow_down':
                            # 增加轮询间隔
                            time.sleep(interval)
                            continue
                        elif error == 'expired_token':
                            print("\n\n[失败] 设备代码已过期")
                            return False
                        elif error == 'access_denied':
                            print("\n\n[失败] 用户拒绝了授权")
                            return False
                        else:
                            print(f"\n\n[失败] 错误: {error}")
                            return False

                print("\n\n[失败] 等待超时")
                return False

            else:
                print(f"[失败] 请求设备代码失败: {response.status_code}")
                print(f"响应: {response.text}")
                return False

    except KeyboardInterrupt:
        print("\n\n[取消] 用户取消了认证")
        return False
    except Exception as e:
        print(f"\n[失败] 认证错误: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    try:
        success = authenticate_earthscope_manual()

        if success:
            print("\n✓ 认证成功，可以开始下载数据")
        else:
            print("\n✗ 认证失败，请重试或查看错误信息")

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
