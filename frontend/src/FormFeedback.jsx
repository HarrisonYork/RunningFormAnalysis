import React from 'react';

const FormFeedback = ({ confidences }) => {
    const THRESHOLD = 70.0;

    const feedbackDatabase = [
        {
            key: 'heel_strike',
            title: 'Heel Strike',
            score: confidences.heel_strike,
            advice: 'You are landing too far back on your heel. Try taking slightly shorter, quicker steps (increasing your cadence) to encourage a midfoot strike directly under your hips.',
            alertColor: '#dc3545'
        },
        {
            key: 'lean_forward',
            title: 'Excessive Forward Lean',
            score: confidences.lean_forward,
            advice: 'You are leaning too far forward from the waist. Focus on running tall, engaging your core, and leading with your chest.',
            alertColor: '#ffc107'
        },
        {
            key: 'arms_tight',
            title: 'Arms Too Tight',
            score: confidences.arms_tight,
            advice: 'Your arms are pinned too tightly to your chest. Relax your shoulders, drop your hands slightly, and let them swing naturally.',
            alertColor: '#fd7e14'
        },
        {
            key: 'arms_loose',
            title: 'Arms Too Loose',
            score: confidences.arms_loose,
            advice: 'Your arms are too lose. Focus on keeping your elbows bent at ~90 degrees and driving them straight back like a pendulum.',
            alertColor: '#fd7e14'
        }
    ];

    const flaggedIssues = feedbackDatabase.filter(issue => issue.score >= THRESHOLD);

    if (flaggedIssues.length === 0) {
        return (
            <div style={{ padding: '1rem'}}>
                <h2 style={{ color: '#ffffff', marginTop: 0 }}>Excellent Form</h2>
                <p style={{ color: '#ffffff', marginBottom: 0 }}>Form analysis did not detect any major biomechanical errors.</p>
            </div>
        );
    }

    return (
        <div className="coaching-feedback">
            <h2 style={{ color: '#ffffff' }}>Form Analysis</h2>
            {flaggedIssues.map((issue) => (
                <div 
                    key={issue.key} 
                    style={{ 
                        borderLeft: `5px solid ${issue.alertColor}`, 
                        backgroundColor: '#ffffff0c', 
                        padding: '15px', 
                        margin: '15px 0 0 0', 
                        borderRadius: '4px',
                        boxShadow: '0 2px 4px rgba(0,0,0,0.05)'
                    }}
                >
                    <h5 style={{ margin: '0 0 8px 0', display: 'flex', justifyContent: 'space-between' }}>
                        <span style={{ lineHeight: '1.5', color: '#ffffff'}}>{issue.title}</span>
                        <span style={{ color: '#ffffff', fontSize: '0.9em' }}>{issue.score}% Confidence</span>
                    </h5>
                    <p style={{ margin: 0, lineHeight: '1.5', textAlign: 'left' }}>
                        {issue.advice}
                    </p>
                </div>
            ))}
        </div>
    );
};

export default FormFeedback;