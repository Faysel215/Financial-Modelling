import React, { useState, useEffect } from 'react';
import { LineChart, AreaChart, Line, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';

const API_BASE_URL = 'http://localhost:5001/api';

const fetchData = async (endpoint) => {
    try {
        const response = await fetch(`${API_BASE_URL}${endpoint}`);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        return await response.json();
    } catch (error) {
        console.error(`Error fetching from ${endpoint}:`, error);
        return [];
    }
};

// --- Reusable Chart Component for Time-Series Data ---
const TimeSeriesChart = ({ data, dataKey, name, color, yAxisFormatter }) => (
    <ResponsiveContainer width="100%" height={250}>
        <AreaChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <defs>
                <linearGradient id={`color-${dataKey}`} x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={color} stopOpacity={0.8}/>
                    <stop offset="95%" stopColor={color} stopOpacity={0}/>
                </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#444" />
            <XAxis dataKey="time" stroke="#ccc" />
            <YAxis stroke="#ccc" tickFormatter={yAxisFormatter} domain={['auto', 'auto']} />
            <Tooltip
                contentStyle={{ backgroundColor: '#222', border: '1px solid #555' }}
                labelStyle={{ color: '#fff' }}
                formatter={yAxisFormatter ? (value) => yAxisFormatter(value) : (value) => value}
            />
            <Area type="monotone" dataKey={dataKey} name={name} stroke={color} fill={`url(#color-${dataKey})`} />
        </AreaChart>
    </ResponsiveContainer>
);

// --- Main App Component ---
export default function App() {
    const [metricsData, setMetricsData] = useState([]);
    const [volumeData, setVolumeData] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const loadAllData = async () => {
            setLoading(true);
            const [metrics, volume] = await Promise.all([
                fetchData('/metrics/1s'),
                fetchData('/volume/executed')
            ]);
            
            const formattedMetrics = metrics.map(item => ({
                ...item,
                time: new Date(item.second).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
            }));
            
            setMetricsData(formattedMetrics);
            setVolumeData(volume);
            setLoading(false);
        };
        loadAllData();
    }, []);

    return (
        <>
            <style>{`
                :root {
                    --bg-color: #1a1a1a; --card-bg: #242424; --text-color: #e0e0e0;
                    --header-color: #ffffff; --border-color: #3a3a3a;
                    --blue: #5e8cff; --green: #52b788; --purple: #b392f0; --orange: #f7b84b;
                }
                body {
                    background-color: var(--bg-color); color: var(--text-color);
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    margin: 0; padding: 20px;
                }
                .dashboard-header { text-align: center; color: var(--header-color); padding-bottom: 10px; border-bottom: 1px solid var(--border-color); }
                .dashboard-header h1 { margin: 0; font-weight: 500; }
                .dashboard-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 20px; margin-top: 20px; }
                .card { background-color: var(--card-bg); border: 1px solid var(--border-color); border-radius: 8px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
                .card h2 { margin-top: 0; font-size: 1.2em; color: var(--header-color); border-bottom: 1px solid var(--border-color); padding-bottom: 10px; margin-bottom: 20px; font-weight: 400; }
                .loading { display: flex; justify-content: center; align-items: center; height: 300px; font-style: italic; color: #888; }
            `}</style>
            
            <div className="dashboard">
                <header className="dashboard-header">
                    <h1>LOBSTER Data Analysis: SPY</h1>
                </header>

                {loading ? <div className="loading">Loading Dashboard Data...</div> : (
                    <div className="dashboard-grid">
                        <div className="card">
                            <h2>Mid-Price (1s Avg)</h2>
                            <TimeSeriesChart data={metricsData} dataKey="avg_mid_price" name="Mid-Price" color="var(--blue)" yAxisFormatter={val => `$${val.toFixed(2)}`} />
                        </div>
                        <div className="card">
                            <h2>Bid-Ask Spread (1s Avg)</h2>
                            <TimeSeriesChart data={metricsData} dataKey="avg_spread" name="Spread" color="var(--orange)" yAxisFormatter={val => `$${val.toFixed(4)}`} />
                        </div>
                        <div className="card">
                            <h2>Market Depth (L5 Vol, 1s Avg)</h2>
                            <TimeSeriesChart data={metricsData} dataKey="avg_market_depth" name="Depth" color="var(--green)" yAxisFormatter={val => val.toLocaleString()} />
                        </div>
                        <div className="card">
                            <h2>Order Book Imbalance (OBI, 1s Avg)</h2>
                            <TimeSeriesChart data={metricsData} dataKey="avg_obi" name="OBI" color="var(--purple)" yAxisFormatter={val => val.toFixed(3)} />
                        </div>
                         <div className="card">
                            <h2>Total Executed Volume</h2>
                             <ResponsiveContainer width="100%" height={250}>
                                <BarChart data={volumeData} margin={{ top: 5, right: 30, left: 100, bottom: 5 }}>
                                     <CartesianGrid strokeDasharray="3 3" stroke="#444"/>
                                     <XAxis type="number" stroke="#ccc" tickFormatter={val => `${val/1000}k`} />
                                     <YAxis type="category" dataKey="trade_type" stroke="#ccc" width={150} />
                                     <Tooltip cursor={{fill: '#444'}} contentStyle={{ backgroundColor: '#222', border: '1px solid #555' }}/>
                                     <Bar dataKey="total_executed_volume" name="Total Volume" fill="var(--green)" />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                )}
            </div>
        </>
    );
}
