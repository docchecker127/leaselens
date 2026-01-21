import streamlit as st
import cv2
import numpy as np
import fitz  # PyMuPDF
import sys
import pytesseract
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Detect if running on Cloud (Linux) or Local (Windows)
if sys.platform.startswith('linux'):
    pytesseract.pytesseract.tesseract_cmd = 'tesseract'
else:
    # 🔴 VERIFY THIS PATH matches your local installation
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ==========================================
# 2. VISION ENGINE (Updated for Initials)
# ==========================================
def check_initials_engine(img_array):
    h, w = img_array.shape
    
    # --- A. COORDINATES: BOTTOM RIGHT CORNER ---
    # We scan the Bottom 15% and Right 20% of the page.
    y_start, y_end = int(h * 0.85), int(h * 0.98) 
    x_start, x_end = int(w * 0.80), int(w * 0.98)
    
    roi = img_array[y_start:y_end, x_start:x_end]
    
    # --- B. PRE-PROCESSING ---
    # Thresholding to find ink (Text becomes White, Paper becomes Black in binary)
    thresh = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 21, 10)
    
    # Remove lines (tables/borders) - Kernel is smaller (20) because initials are small
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    remove_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
    clean_roi = cv2.subtract(thresh, remove_lines)
    
    contours, _ = cv2.findContours(clean_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Debug Image (Color)
    debug_img = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    initials_found = False
    
    for cnt in contours:
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        
        # --- C. THE "INK DENSITY" CHECK (The Fix) ---
        # We check the ORIGINAL grayscale pixels to see if this is a solid block (Logo/Header)
        # or a scribbly line (Initial).
        
        roi_chunk = roi[y:y+h_box, x:x+w_box]
        total_pixels = w_box * h_box
        
        if total_pixels > 0:
            # Count pixels darker than 100 (Ink)
            dark_pixels = np.count_nonzero(roi_chunk < 100)
            fill_ratio = dark_pixels / total_pixels
        else:
            fill_ratio = 0
            
        # --- D. FILTER LOGIC ---
        
        # 1. Size Check
        # Initials are small (>15px) but NOT huge (>200px would be a logo/picture)
        is_visible = (w_box > 15 and h_box > 15)
        is_not_huge = (w_box < 200 and h_box < 150)
        
        # 2. Density Check
        # Initials are scribbles (< 25% ink).
        # Logos & Headers are blocks (> 30% ink).
        is_ink_sparse = (fill_ratio < 0.25)
        
        # 3. Aspect Ratio Check
        # Initials are rarely perfectly square or super wide rectangles
        if h_box > 0: ratio = w_box / h_box
        else: ratio = 0
        is_not_bar = (ratio < 5.0) # Rejects long horizontal bars
        
        if is_visible and is_not_huge and is_ink_sparse and is_not_bar:
            # ✅ GREEN BOX (Actual Initials)
            cv2.rectangle(debug_img, (x, y), (x+w_box, y+h_box), (0, 255, 0), 2)
            initials_found = True
        else:
            # 🔴 RED BOX (Logos, Headers, Noise) - Shown for debugging
            cv2.rectangle(debug_img, (x, y), (x+w_box, y+h_box), (0, 0, 255), 1)
            
    return initials_found, debug_img

# ==========================================
# 3. PDF PROCESSOR (Cached)
# ==========================================
@st.cache_data(show_spinner=False)
def process_lease(file_bytes):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    results = []
    
    for i in range(len(doc)):
        page = doc[i]
        
        # Convert to Image
        pix = page.get_pixmap(dpi=150)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Run Vision Engine
        found, debug_img = check_initials_engine(gray)
        
        results.append({
            "page_num": i + 1,
            "has_initials": found,
            "debug_img": debug_img
        })
    
    return results

# ==========================================
# 4. STREAMLIT UI
# ==========================================
st.set_page_config(page_title="LeaseLens - Audit Tool", page_icon="🏢", layout="wide")

st.title("🏢 LeaseLens Audit")
st.markdown("""
**Property Managers:** Drag & Drop a lease file (or a whole folder combined).
We scan the **Bottom-Right Corner** of every page for missing initials.
""")

uploaded_file = st.file_uploader("📂 Upload Lease PDF", type=["pdf"])

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    
    with st.spinner("Auditing Lease File..."):
        page_results = process_lease(file_bytes)
    
    missed_count = 0
    
    # SUMMARY DASHBOARD
    st.divider()
    
    for res in page_results:
        p_num = res["page_num"]
        has_initials = res["has_initials"]
        
        if has_initials:
            status = "✅ INITIALS FOUND"
            color = "green"
            expand = False
        else:
            status = "❌ MISSING INITIALS"
            color = "red"
            expand = True # Auto-open the missing ones
            missed_count += 1
            
        with st.expander(f"Page {p_num}: {status}", expanded=expand):
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown(f"### Status: :{color}[{status}]")
                if not has_initials:
                    st.error("⚠️ Action: Check Bottom Right Corner.")
            with col2:
                # Show the bottom-right crop only
                st.image(res["debug_img"], channels="BGR", caption=f"Bottom Right of Page {p_num}")

    st.divider()
    if missed_count == 0:
        st.success("🎉 Perfect Lease! All pages appear initialed.")
    else:
        st.error(f"🚨 FOUND {missed_count} PAGES MISSING INITIALS.")
