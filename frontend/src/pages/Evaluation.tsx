import React, { useState, useEffect } from 'react';
import api from '../api';
import { Beaker, Search, BarChart3, AlertCircle } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

export default function Evaluation() {
  const [experiments, setExperiments] = useState<any[]>([]);
  const [selectedExp, setSelectedExp] = useState<string>('');
  const [evalResults, setEvalResults] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    api.get('/experiments').then(res => {
      setExperiments(res.data.experiments);
      if (res.data.experiments.length > 0) {
        setSelectedExp(res.data.experiments[0].experiment_id);
      }
    }).catch(console.error);
  }, []);

  useEffect(() => {
    if (selectedExp) {
      setLoading(true);
      api.get(`/evaluation/${selectedExp}`)
        .then(res => {
          setEvalResults(res.data);
          setLoading(false);
        })
        .catch(err => {
          // If 404, it means it hasn't been evaluated yet
          setEvalResults(null);
          setLoading(false);
        });
    }
  }, [selectedExp]);

  const handleRunEvaluation = () => {
    if (!selectedExp) return;
    setLoading(true);
    api.post('/evaluation/run', { experiment_id: selectedExp })
      .then(() => {
        // Poll for results
        const interval = setInterval(() => {
          api.get(`/evaluation/${selectedExp}`)
            .then(res => {
              setEvalResults(res.data);
              setLoading(false);
              clearInterval(interval);
            })
            .catch(() => {});
        }, 2000);
      })
      .catch(console.error);
  };

  const expDetails = experiments.find(e => e.experiment_id === selectedExp);
  
  // Prepare chart data for shots
  const chartData = [];
  if (evalResults?.shot_results) {
    for (const [shot, result] of Object.entries(evalResults.shot_results) as [string, any][]) {
      chartData.push({
        name: `${shot}-Shot`,
        accuracy: parseFloat((result.mean_accuracy * 100).toFixed(1)),
        f1: parseFloat((result.mean_f1 * 100).toFixed(1)),
      });
    }
  }

  return (
    <div className="animate-fade-in">
      <div className="mb-8">
        <h1 className="text-2xl font-bold mb-2">Scientific Evaluation</h1>
        <p className="text-secondary">Evaluate the trained model's ability to adapt to the held-out unseen class.</p>
      </div>

      <div className="card mb-8">
        <div className="flex items-center gap-4">
          <div className="flex-1 form-group mb-0">
            <label className="form-label">Select Trained Model (Experiment)</label>
            <select 
              className="form-control"
              value={selectedExp}
              onChange={e => setSelectedExp(e.target.value)}
            >
              <option value="">-- Select Experiment --</option>
              {experiments.map(exp => (
                <option key={exp.experiment_id} value={exp.experiment_id}>
                  {exp.experiment_id} - Held Out: {exp.held_out_class} 
                  ({new Date(exp.timestamp).toLocaleDateString()})
                </option>
              ))}
            </select>
          </div>
          
          <div className="pt-6">
            <button 
              className="btn btn-primary" 
              onClick={handleRunEvaluation}
              disabled={!selectedExp || loading}
            >
              {loading ? 'Running...' : 'Run N-Shot Evaluation'}
            </button>
          </div>
        </div>
      </div>

      {!evalResults && !loading && selectedExp && (
        <div className="card border-l-4 border-l-warning flex items-center gap-4 p-6">
          <AlertCircle className="text-warning" size={32} />
          <div>
            <h3 className="font-semibold text-lg text-warning mb-1">Not Evaluated Yet</h3>
            <p className="text-secondary">This model has not been scientifically evaluated on its held-out class. Click the button above to run the evaluation.</p>
          </div>
        </div>
      )}

      {loading && (
        <div className="flex justify-center items-center h-48">
          <div className="animate-pulse text-primary flex flex-col items-center gap-4">
            <Beaker size={48} className="animate-bounce" />
            <span>Running multiple few-shot evaluation episodes...</span>
          </div>
        </div>
      )}

      {evalResults && !loading && (
        <>
          <div className="card mb-8 bg-gradient-to-br from-[var(--bg-surface)] to-[var(--bg-surface-elevated)] border-primary">
            <div className="card-header border-b border-[var(--border-light)] pb-4 mb-6">
              <h2 className="card-title text-primary">Evaluation Report</h2>
              <span className="text-sm text-secondary">Unseen Class: <strong className="text-white">{evalResults.held_out_class}</strong></span>
            </div>

            {chartData.length > 0 && (
              <div className="h-72 mb-8">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData} margin={{ top: 20, right: 30, left: 0, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" vertical={false} />
                    <XAxis dataKey="name" stroke="#94a3b8" />
                    <YAxis stroke="#94a3b8" domain={[0, 100]} unit="%" />
                    <Tooltip contentStyle={{ backgroundColor: '#1a1f2e', borderColor: '#2d354a', color: '#fff' }} cursor={{fill: 'rgba(255,255,255,0.05)'}} />
                    <Legend />
                    <Bar dataKey="accuracy" name="Accuracy (%)" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="f1" name="F1 Score (%)" fill="#10b981" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}

            <div className="table-container">
              <table className="table">
                <thead>
                  <tr>
                    <th>Protocol</th>
                    <th>Mean Accuracy (95% CI)</th>
                    <th>Macro F1</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>Adaptation Time</th>
                  </tr>
                </thead>
                <tbody>
                  {['1', '3', '5', '10'].map(shot => {
                    const res = evalResults.shot_results[shot];
                    if (!res) return null;
                    return (
                      <tr key={shot}>
                        <td className="font-semibold text-primary">{shot}-Shot</td>
                        <td>
                          <span className="font-bold text-white">{(res.mean_accuracy * 100).toFixed(1)}%</span>
                          <span className="text-secondary text-xs ml-2">±{(res.ci_95_accuracy * 100).toFixed(2)}</span>
                        </td>
                        <td>{(res.mean_f1 * 100).toFixed(1)}%</td>
                        <td>{(res.mean_precision * 100).toFixed(1)}%</td>
                        <td>{(res.mean_recall * 100).toFixed(1)}%</td>
                        <td className="text-secondary">{res.mean_adaptation_time_ms} ms</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            
            <div className="mt-6 p-4 bg-[var(--bg-main)] rounded text-sm text-secondary border border-[var(--border-light)]">
              <strong>Methodology Note:</strong> For each shot configuration, {evalResults.shot_results['5']?.num_episodes || 50} independent episodes were sampled. In each episode, K support images were sampled from the unseen class, and disjoint query images were evaluated against the constructed prototype and known class prototypes.
            </div>
          </div>
        </>
      )}
    </div>
  );
}
