import React, { useState, useEffect } from 'react';
import api from '../api';
import { Upload, Plus, Image as ImageIcon, Zap, CheckCircle2 } from 'lucide-react';

export default function Predict() {
  const [classes, setClasses] = useState<any[]>([]);
  const [newClassName, setNewClassName] = useState('');
  const [modelLoaded, setModelLoaded] = useState(false);
  const [loading, setLoading] = useState(false);
  
  // Prediction state
  const [queryImage, setQueryImage] = useState<File | null>(null);
  const [queryPreview, setQueryPreview] = useState<string>('');
  const [prediction, setPrediction] = useState<any>(null);

  useEffect(() => {
    fetchClasses();
  }, []);

  const fetchClasses = () => {
    api.get('/adaptation/classes').then(res => {
      setClasses(res.data.classes);
      setModelLoaded(res.data.model_loaded);
    }).catch(console.error);
  };

  const handleCreateClass = () => {
    if (!newClassName.trim()) return;
    setLoading(true);
    api.post('/adaptation/create-class', { class_name: newClassName })
      .then(() => {
        setNewClassName('');
        fetchClasses();
        setLoading(false);
      })
      .catch(err => {
        console.error(err);
        setLoading(false);
      });
  };

  const handleSupportUpload = (classId: string, e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    
    setLoading(true);
    const formData = new FormData();
    formData.append('file', e.target.files[0]);
    
    api.post(`/adaptation/${classId}/add-support`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    })
    .then(() => {
      fetchClasses();
      setLoading(false);
    })
    .catch(err => {
      console.error(err);
      setLoading(false);
    });
  };

  const handleBuildPrototype = (classId: string) => {
    setLoading(true);
    api.post(`/adaptation/${classId}/build-prototype`)
      .then(() => {
        fetchClasses();
        setLoading(false);
      })
      .catch(err => {
        console.error(err);
        setLoading(false);
      });
  };

  const handleQueryUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    
    const file = e.target.files[0];
    setQueryImage(file);
    setQueryPreview(URL.createObjectURL(file));
    setPrediction(null);
  };

  const handlePredict = () => {
    if (!queryImage) return;
    
    setLoading(true);
    const formData = new FormData();
    formData.append('file', queryImage);
    
    // Only predict against ready classes
    const readyClassIds = classes.filter(c => c.is_ready).map(c => c.class_id);
    formData.append('class_ids', JSON.stringify(readyClassIds));
    
    api.post('/prediction', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    })
    .then(res => {
      setPrediction(res.data);
      setLoading(false);
    })
    .catch(err => {
      console.error(err);
      setLoading(false);
    });
  };

  if (!modelLoaded) {
    return (
      <div className="flex flex-col justify-center items-center h-64 text-center">
        <Zap size={48} className="text-warning mb-4" />
        <h2 className="text-xl font-bold mb-2">No Model Loaded</h2>
        <p className="text-secondary max-w-md">
          To use few-shot prediction, you must first train a model or load a checkpoint from the Evaluation page. The inference engine needs a trained Prototypical Network.
        </p>
      </div>
    );
  }

  const readyClasses = classes.filter(c => c.is_ready);

  return (
    <div className="animate-fade-in">
      <div className="mb-8">
        <h1 className="text-2xl font-bold mb-2">Few-Shot Adaptation & Inference</h1>
        <p className="text-secondary">Dynamically introduce completely new classes using a few labeled support images without retraining the model.</p>
      </div>

      <div className="grid grid-cols-2 gap-8">
        {/* Left Column: Class Management */}
        <div className="flex flex-col gap-6">
          <div className="card">
            <div className="card-header">
              <h2 className="card-title text-primary">1. Create New Classes</h2>
            </div>
            
            <div className="flex gap-2 mb-6">
              <input 
                type="text" 
                className="form-control" 
                placeholder="New breed name (e.g., Hereford)" 
                value={newClassName}
                onChange={e => setNewClassName(e.target.value)}
              />
              <button className="btn btn-primary whitespace-nowrap" onClick={handleCreateClass} disabled={loading || !newClassName}>
                <Plus size={16} /> Add Class
              </button>
            </div>

            <div className="flex flex-col gap-4">
              {classes.length === 0 ? (
                <div className="text-center p-4 text-muted border border-dashed border-[var(--border-light)] rounded-md">
                  No temporary classes created yet.
                </div>
              ) : (
                classes.map(cls => (
                  <div key={cls.class_id} className={`p-4 rounded-md border ${cls.is_ready ? 'border-success bg-[var(--success-bg)]' : 'border-[var(--border-light)] bg-[var(--bg-main)]'}`}>
                    <div className="flex justify-between items-center mb-3">
                      <h3 className="font-semibold text-lg flex items-center gap-2">
                        {cls.class_name}
                        {cls.is_ready && <CheckCircle2 size={16} className="text-success" />}
                      </h3>
                      <span className="badge badge-secondary">{cls.k_shot}-Shot</span>
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <label className="btn btn-secondary text-xs cursor-pointer">
                          <Upload size={14} /> Upload Support
                          <input type="file" className="hidden" accept="image/*" onChange={(e) => handleSupportUpload(cls.class_id, e)} disabled={loading} />
                        </label>
                      </div>
                      
                      <button 
                        className={`btn text-xs ${cls.k_shot > 0 && !cls.is_ready ? 'btn-primary' : 'btn-secondary opacity-50'}`}
                        onClick={() => handleBuildPrototype(cls.class_id)}
                        disabled={loading || cls.k_shot === 0 || cls.is_ready}
                      >
                        <Zap size={14} /> Build Prototype
                      </button>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>

        {/* Right Column: Prediction */}
        <div className="flex flex-col gap-6">
          <div className="card opacity-100 transition-opacity" style={{ opacity: readyClasses.length > 0 ? 1 : 0.5 }}>
            <div className="card-header">
              <h2 className="card-title text-success">2. Predict Query Image</h2>
            </div>
            
            {readyClasses.length === 0 ? (
              <div className="p-4 text-center text-warning bg-warning/10 rounded-md border border-warning/20">
                Create and build a prototype for at least one class first.
              </div>
            ) : (
              <>
                <div className="flex flex-col items-center justify-center p-6 border-2 border-dashed border-[var(--border-light)] rounded-lg mb-4 bg-[var(--bg-main)] relative overflow-hidden">
                  {queryPreview ? (
                    <img src={queryPreview} alt="Query" className="h-48 object-contain z-10" />
                  ) : (
                    <div className="flex flex-col items-center text-muted">
                      <ImageIcon size={48} className="mb-2 opacity-50" />
                      <span>Upload query image to classify</span>
                    </div>
                  )}
                  
                  <input 
                    type="file" 
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-20" 
                    accept="image/*" 
                    onChange={handleQueryUpload}
                  />
                </div>
                
                <button 
                  className="btn btn-primary w-full py-3 text-lg font-bold shadow-lg shadow-primary/30"
                  onClick={handlePredict}
                  disabled={loading || !queryImage}
                >
                  <Zap size={20} /> Classify Image
                </button>
              </>
            )}
          </div>

          {prediction && (
            <div className="card border-primary animate-fade-in bg-gradient-to-br from-[var(--bg-surface)] to-[#1e3a8a]/20">
              <h3 className="text-secondary text-sm mb-1 uppercase tracking-wider font-semibold">Prediction Result</h3>
              <div className="text-3xl font-bold text-primary mb-2">{prediction.predicted_class}</div>
              
              <div className="w-full bg-[var(--bg-main)] h-4 rounded-full overflow-hidden mb-1 border border-[var(--border-light)]">
                <div 
                  className="h-full bg-primary" 
                  style={{ width: `${prediction.confidence * 100}%` }}
                ></div>
              </div>
              <div className="flex justify-between text-xs text-secondary mb-6">
                <span>Confidence</span>
                <span>{(prediction.confidence * 100).toFixed(2)}%</span>
              </div>
              
              <h4 className="text-sm font-semibold mb-3 border-b border-[var(--border-light)] pb-2">Class Similarities</h4>
              <div className="flex flex-col gap-3">
                {prediction.all_classes.sort((a,b) => b.probability - a.probability).map((c: any) => (
                  <div key={c.class_id}>
                    <div className="flex justify-between text-sm mb-1">
                      <span>{c.class_name}</span>
                      <span className="text-secondary">{(c.probability * 100).toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-[var(--bg-main)] h-1.5 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-secondary" 
                        style={{ width: `${c.probability * 100}%` }}
                      ></div>
                    </div>
                  </div>
                ))}
              </div>
              
              <div className="text-xs text-muted text-right mt-4 pt-4 border-t border-[var(--border-light)]">
                Inference time: {prediction.inference_time_ms}ms
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
