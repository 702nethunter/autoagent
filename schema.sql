-- =============================================================
-- Generative Dev Agents — SQL Server Schema
-- Based on: Park et al. 2023 "Generative Agents" (arXiv:2304.03442)
-- =============================================================

-- ── Agents ────────────────────────────────────────────────────
CREATE TABLE Agents (
    agent_id        INT IDENTITY(1,1) PRIMARY KEY,
    name            NVARCHAR(100)  NOT NULL,
    role            NVARCHAR(50)   NOT NULL,   -- 'pm', 'dotnet', 'cpp'
    persona         NVARCHAR(MAX)  NOT NULL,   -- natural language persona description
    created_at      DATETIME2      NOT NULL DEFAULT GETUTCDATE()
);

-- ── Memory Stream ─────────────────────────────────────────────
-- Each row is one memory object (observation / reflection / plan)
CREATE TABLE MemoryStream (
    memory_id       INT IDENTITY(1,1) PRIMARY KEY,
    agent_id        INT            NOT NULL REFERENCES Agents(agent_id),
    memory_type     NVARCHAR(20)   NOT NULL,   -- 'observation' | 'reflection' | 'plan'
    description     NVARCHAR(MAX)  NOT NULL,   -- natural language
    importance      FLOAT          NOT NULL DEFAULT 5.0,  -- LLM-rated 1–10 poignancy
    embedding_json  NVARCHAR(MAX)  NULL,        -- JSON float array for cosine similarity
    created_at      DATETIME2      NOT NULL DEFAULT GETUTCDATE(),
    last_accessed   DATETIME2      NOT NULL DEFAULT GETUTCDATE(),
    -- Soft-delete so reflections can reference originals
    is_active       BIT            NOT NULL DEFAULT 1
);
CREATE INDEX IX_Memory_Agent ON MemoryStream(agent_id, is_active, created_at DESC);

-- ── Plans ─────────────────────────────────────────────────────
-- Hierarchical: sprint → task → subtask (parent_plan_id NULL = top-level)
CREATE TABLE Plans (
    plan_id         INT IDENTITY(1,1) PRIMARY KEY,
    agent_id        INT            NOT NULL REFERENCES Agents(agent_id),
    parent_plan_id  INT            NULL     REFERENCES Plans(plan_id),
    assigned_to     INT            NULL     REFERENCES Agents(agent_id),  -- dev assigned
    title           NVARCHAR(500)  NOT NULL,
    description     NVARCHAR(MAX)  NULL,
    status          NVARCHAR(20)   NOT NULL DEFAULT 'pending',  -- pending|in_progress|done|blocked
    priority        INT            NOT NULL DEFAULT 5,           -- 1 (highest) – 10
    created_at      DATETIME2      NOT NULL DEFAULT GETUTCDATE(),
    started_at      DATETIME2      NULL,
    completed_at    DATETIME2      NULL
);
CREATE INDEX IX_Plans_Agent   ON Plans(agent_id, status);
CREATE INDEX IX_Plans_Assigned ON Plans(assigned_to, status);

-- ── Messages (agent-to-agent communication) ───────────────────
CREATE TABLE Messages (
    message_id      INT IDENTITY(1,1) PRIMARY KEY,
    from_agent_id   INT            NOT NULL REFERENCES Agents(agent_id),
    to_agent_id     INT            NOT NULL REFERENCES Agents(agent_id),
    content         NVARCHAR(MAX)  NOT NULL,
    message_type    NVARCHAR(30)   NOT NULL DEFAULT 'chat',  -- chat|standup|assignment|escalation
    is_read         BIT            NOT NULL DEFAULT 0,
    created_at      DATETIME2      NOT NULL DEFAULT GETUTCDATE()
);
CREATE INDEX IX_Messages_To ON Messages(to_agent_id, is_read, created_at);

-- ── Incidents ─────────────────────────────────────────────────
-- Logged when agents encounter blockers, bugs, or failures
CREATE TABLE Incidents (
    incident_id     INT IDENTITY(1,1) PRIMARY KEY,
    reported_by     INT            NOT NULL REFERENCES Agents(agent_id),
    assigned_to     INT            NULL     REFERENCES Agents(agent_id),
    plan_id         INT            NULL     REFERENCES Plans(plan_id),
    title           NVARCHAR(500)  NOT NULL,
    description     NVARCHAR(MAX)  NOT NULL,
    severity        NVARCHAR(10)   NOT NULL DEFAULT 'medium',  -- low|medium|high|critical
    status          NVARCHAR(20)   NOT NULL DEFAULT 'open',    -- open|investigating|resolved|closed
    root_cause      NVARCHAR(MAX)  NULL,
    resolution      NVARCHAR(MAX)  NULL,
    created_at      DATETIME2      NOT NULL DEFAULT GETUTCDATE(),
    resolved_at     DATETIME2      NULL
);
CREATE INDEX IX_Incidents_Status ON Incidents(status, severity, created_at DESC);

-- ── Reflections link table (which memories formed a reflection) ─
CREATE TABLE ReflectionSources (
    reflection_memory_id  INT NOT NULL REFERENCES MemoryStream(memory_id),
    source_memory_id      INT NOT NULL REFERENCES MemoryStream(memory_id),
    PRIMARY KEY (reflection_memory_id, source_memory_id)
);
