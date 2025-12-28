from enum import StrEnum


class NodeType(StrEnum):
    Process_State = "Process_State"
    Digital_System = "Digital_System"
    Digital_Data_Object = "Digital_Data_Object"
    Physical_Raw_Material = "Physical_Raw_Material"
    Physical_Processed_Artifact = "Physical_Processed_Artifact"


class EdgeType(StrEnum):
    # 物理的取得
    Physical_Possess_PreExistent = "Physical_Possess_PreExistent"
    Physical_Obtain_Required_Item = "Physical_Obtain_Required_Item"
    Physical_Obtain_Raw_Material = "Physical_Obtain_Raw_Material"

    # 物理的変換
    Physical_Print_From_Digital = "Physical_Print_From_Digital"
    Physical_Write_or_Process = "Physical_Write_or_Process"
    Physical_Duplicate_Copy = "Physical_Duplicate_Copy"
    Physical_Combine_or_Package = "Physical_Combine_or_Package"

    # 物理的提出
    Physical_Submit_To_Window = "Physical_Submit_To_Window"
    Physical_Submit_Via_Mail = "Physical_Submit_Via_Mail"

    # デジタル
    Digital_Access_Website = "Digital_Access_Website"
    Digital_Download_File = "Digital_Download_File"
    Digital_Authenticate_User = "Digital_Authenticate_User"
    Digital_Enter_Text_Data = "Digital_Enter_Text_Data"
    Digital_Photograph_Document = "Digital_Photograph_Document"
    Digital_Upload_File = "Digital_Upload_File"
    Digital_Submit_Data = "Digital_Submit_Data"
    System_Auto_Link_Data = "System_Auto_Link_Data"

    # 待機
    Time_Wait_For_Processing = "Time_Wait_For_Processing"


# --- 2. プロンプト生成用の説明文 ---
# Enumをキーにして記述します

NODE_DESCRIPTIONS = {
    NodeType.Process_State: "プロセスの始点、終点、待機状態。(例: 開始(自宅), 申請完了, 認定通知, 審査待ち)",
    NodeType.Digital_System: "ユーザーが操作するWebサイトやアプリの画面。(例: マイナポータル, 自治体HP, ログイン画面)",
    NodeType.Digital_Data_Object: "デジタルデータやファイル。(例: 申請書PDF, 入力データ, 撮影画像, 電子署名済データ)",
    NodeType.Physical_Raw_Material: "まだ情報の入っていない物理的な素材。(例: 未記入の申請書用紙, 空の封筒, 切手)",
    NodeType.Physical_Processed_Artifact: "情報が記入された書類、または意味を持つ物理的成果物。(例: 記入済申請書, 健康保険証(原本), 封入済封筒, コピーした書類)",
}

EDGE_DEFINITIONS = {
    EdgeType.Physical_Possess_PreExistent: {
        "description": "手続き開始時点で既に所持しているもの。コスト0。(例: 免許証、通帳、家にあった封筒)",
        "source": NodeType.Process_State,
        "target": [NodeType.Physical_Processed_Artifact, NodeType.Physical_Raw_Material],
        "meta": {"base_cost": 0, "color": "#B2BEC3", "category": "possession"},
    },
    EdgeType.Physical_Obtain_Required_Item: {
        "description": "役所や外部機関へ行き、完成された証明書や物品を入手する。(例: 住民票の取得、母子手帳の交付)",
        "source": NodeType.Process_State,
        "target": NodeType.Physical_Processed_Artifact,
        "meta": {"base_cost": 15, "color": "#FF6B6B", "category": "physical_acquire"},
    },
    EdgeType.Physical_Obtain_Raw_Material: {
        "description": "役所や店へ行き、未記入の用紙や資材を入手する。(例: 窓口で申請書をもらう, コンビニで切手・封筒を買う)",
        "source": NodeType.Process_State,
        "target": NodeType.Physical_Raw_Material,
        "meta": {"base_cost": 2, "color": "#FFCDB3", "category": "physical_acquire"},
    },
    EdgeType.Physical_Print_From_Digital: {
        "description": "PDF等のデータを紙に印刷し、物理素材にする。(例: コンビニ印刷)",
        "source": NodeType.Digital_Data_Object,
        "target": NodeType.Physical_Raw_Material,
        "meta": {"base_cost": 5, "color": "#FFD1D1", "category": "physical_transform"},
    },
    EdgeType.Physical_Write_or_Process: {
        "description": "素材や書類に「記入」「押印」「貼付」を行い、提出可能な状態にする。(例: 申請書への記入)",
        "source": [NodeType.Physical_Raw_Material, NodeType.Physical_Processed_Artifact],
        "target": NodeType.Physical_Processed_Artifact,
        "meta": {"base_cost": 15, "color": "#FFA07A", "category": "physical_transform"},
    },
    EdgeType.Physical_Duplicate_Copy: {
        "description": "原本を複写してコピーを作る。(例: コンビニコピー)",
        "source": NodeType.Physical_Processed_Artifact,
        "target": NodeType.Physical_Processed_Artifact,
        "meta": {"base_cost": 5, "color": "#FF8E8E", "category": "physical_transform"},
    },
    EdgeType.Physical_Combine_or_Package: {
        "description": "複数の書類をまとめたり、封筒に入れたりする。(例: 封入、クリップ留め)",
        "source": [NodeType.Physical_Processed_Artifact, NodeType.Physical_Raw_Material],
        "target": NodeType.Physical_Processed_Artifact,
        "meta": {"base_cost": 2, "color": "#FFCDB3", "category": "physical_transform"},
    },
    EdgeType.Physical_Submit_To_Window: {
        "description": "成果物を持って役所窓口へ移動し、提出して完了状態へ遷移する。",
        "source": NodeType.Physical_Processed_Artifact,
        "target": NodeType.Process_State,
        "meta": {"base_cost": 15, "color": "#FF6B6B", "category": "physical_submit"},
    },
    EdgeType.Physical_Submit_Via_Mail: {
        "description": "成果物を持ってポストへ移動し、投函して完了状態へ遷移する。",
        "source": NodeType.Physical_Processed_Artifact,
        "target": NodeType.Process_State,
        "meta": {"base_cost": 10, "color": "#FF8E8E", "category": "physical_submit"},
    },
    EdgeType.Digital_Access_Website: {
        "description": "Webサイトやシステムにアクセスを開始する。",
        "source": NodeType.Process_State,
        "target": NodeType.Digital_System,
        "meta": {"base_cost": 1, "color": "#74B9FF", "category": "digital_action"},
    },
    EdgeType.Digital_Download_File: {
        "description": "Webサイトから様式PDF等をダウンロードしてファイルを得る。",
        "source": NodeType.Digital_System,
        "target": NodeType.Digital_Data_Object,
        "meta": {"base_cost": 2, "color": "#96CEB4", "category": "digital_action"},
    },
    EdgeType.Digital_Authenticate_User: {
        "description": "ログイン、パスワード入力などを行い、システムの権限状態を遷移させる。",
        "source": NodeType.Digital_System,
        "target": NodeType.Digital_System,
        "meta": {"base_cost": 5, "color": "#45B7D1", "category": "digital_action"},
    },
    EdgeType.Digital_Enter_Text_Data: {
        "description": "フォーム入力を行い、入力データを作成する。",
        "source": NodeType.Digital_System,
        "target": NodeType.Digital_Data_Object,
        "meta": {"base_cost": 10, "color": "#4ECDC4", "category": "digital_action"},
    },
    EdgeType.Digital_Photograph_Document: {
        "description": "スマホ等で物理書類を撮影し、画像データ化する。",
        "source": NodeType.Physical_Processed_Artifact,
        "target": NodeType.Digital_Data_Object,
        "meta": {"base_cost": 5, "color": "#FFEAA7", "category": "digital_action"},
    },
    EdgeType.Digital_Upload_File: {
        "description": "データファイルをシステムにアップロードし、添付済みのデータ状態にする。",
        "source": NodeType.Digital_Data_Object,
        "target": NodeType.Digital_Data_Object,
        "meta": {"base_cost": 2, "color": "#96CEB4", "category": "digital_action"},
    },
    EdgeType.Digital_Submit_Data: {
        "description": "入力・添付済みのデータを送信し、完了状態へ遷移する。",
        "source": NodeType.Digital_Data_Object,
        "target": NodeType.Process_State,
        "meta": {"base_cost": 1, "color": "#74B9FF", "category": "digital_action"},
    },
    EdgeType.System_Auto_Link_Data: {
        "description": "マイナ連携や公金受取口座登録により、ユーザー入力なしでデータが自動生成・連携されたとみなす。(コスト0のDX指標)",
        "source": NodeType.Digital_System,
        "target": NodeType.Digital_Data_Object,
        "meta": {"base_cost": 0, "color": "#B2BEC3", "category": "system_action"},
    },
    EdgeType.Time_Wait_For_Processing: {
        "description": "申請完了後、審査や通知の到着を待つ時間。",
        "source": NodeType.Process_State,
        "target": NodeType.Process_State,
        "meta": {"base_cost": 0, "color": "#DFE6E9", "category": "wait"},
    },
}
