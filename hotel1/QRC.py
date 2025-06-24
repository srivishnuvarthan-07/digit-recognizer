import qrcode

# Custom URI
data = "http://192.168.91.39:3000"

# Generate the QR code
qr = qrcode.make(data)

# Save the QR code as an image
qr.save("app_qrcode.png")
print("QR code saved as app_qrcode.png")
