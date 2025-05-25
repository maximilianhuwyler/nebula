// Unlearning Module
const UnlearningManager = (function() {
    const UNLEARNING_METHODS = {
        BASIC_RETRAINING: 'Basic Retraining',
        KNOWLEDGE_DISTILLATION: 'Knowledge Distillation',
        GRADIENT_ASCENT: 'Gradient Ascent',
    };

    const Config = {
        method: UNLEARNING_METHODS.BASIC_RETRAINING,
        leavingNodePercent: 0,
        departureRound: 5,
    };

    function initializeEventListeners() {
        const unlearningMethod = document.getElementById('unlearningMethod');
        const leavingNodePercentInput = document.getElementById('leavingNodePercentInput');
        const leavingNodePercentValue = document.getElementById('leavingNodePercentValue');
        const departureRound = document.getElementById('departureRound');
        const rounds = document.getElementById('rounds');

        unlearningMethod.addEventListener('change', function() {
            Config.method = unlearningMethod.value;
        });

        leavingNodePercentInput.addEventListener('change', function() {
            leavingNodePercentValue.value = leavingNodePercentInput.value;
            Config.leavingNodePercent = leavingNodePercentInput.value;
        });

        leavingNodePercentValue.addEventListener('change', function() {
            leavingNodePercentInput.value = leavingNodePercentValue.value;
            Config.leavingNodePercent = leavingNodePercentValue.value;
        });

        departureRound.addEventListener('change', function() {
            departureRound.value = Math.min(Number(departureRound.value), Number(rounds.value) - 1);
            Config.departureRound = departureRound.value;
        });

        rounds.addEventListener('change', function() {
            departureRound.value = Math.max(0, Math.min(Number(departureRound.value), Number(rounds.value) - 1));
            Config.departureRound = departureRound.value;
        });
    }

    function resetUnlearningConfig() {
        Config.method = UNLEARNING_METHODS.BASIC_RETRAINING;
        document.getElementById("unlearningMethod").value = Config.method;

        Config.leavingNodePercent = 0;
        document.getElementById("leavingNodePercentInput").value = Config.leavingNodePercent;
        document.getElementById("leavingNodePercentValue").value = Config.leavingNodePercent;

        Config.departureRound = 5;
        document.getElementById("departureRound").value = Config.departureRound;
    }

    return {
        UNLEARNING_METHODS,
        initializeEventListeners,
        getUnlearningConfig: () => Config,
        resetUnlearningConfig
    };
})();

export default UnlearningManager;
