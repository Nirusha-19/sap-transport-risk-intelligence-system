def risk_gauge(percent, color):

    return f"""
    <div style="display:flex;justify-content:center;">
        <div style="
            width:160px;
            height:160px;
            border-radius:50%;
            background: conic-gradient({color} {percent}%, #e6e6e6 {percent}%);
            display:flex;
            align-items:center;
            justify-content:center;
            font-size:24px;
            font-weight:bold;
            color:white;
        ">
            {percent}%
        </div>
    </div>
    """